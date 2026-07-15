"""Command-line interface: compute a bWSME conformational landscape for a
PDB structure and write profile/landscape/probability files plus plots,
mirroring the outputs of FesCalc_Block.m / DSCcalc_Block.m."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pipeline import (
    AlanineScanPipelineResult,
    DEFAULT_PH_VALUES,
    PipelineResult,
    run_alanine_scan_pipeline,
    run_pipeline,
    run_pipeline_multi_ph,
)
from .wsme import WSMEParams


def _build_params(args) -> WSMEParams:
    base = WSMEParams.soluble_protein_defaults() if args.preset == "soluble" else WSMEParams()
    overrides = {}
    if args.temp is not None:
        overrides["T"] = args.temp
    if args.ene is not None:
        overrides["ene"] = args.ene
    if args.ds is not None:
        overrides["DS"] = args.ds
    if args.dcp is not None:
        overrides["DCp"] = args.dcp
    if args.ionic_strength is not None:
        overrides["IS"] = args.ionic_strength
    if args.dielectric is not None:
        overrides["dielectric"] = args.dielectric
    return WSMEParams(**{**base.__dict__, **overrides})


def main(argv=None):
    p = argparse.ArgumentParser(prog="wsme-gpcr", description="Compute bWSME conformational free-energy landscapes.")
    p.add_argument("pdb", help="Path to a PDB or mmCIF structure file")
    p.add_argument("--chain", default=None, help="Chain ID to use (default: first chain with standard residues)")
    p.add_argument("--model", type=int, default=0, help="Model index for multi-model files (default: 0)")
    p.add_argument("--ph", type=float, default=7.0, help="pH for charge assignment via continuous Henderson-Hasselbalch titration (default: 7.0; ignored with --all-ph/--ph-values)")
    p.add_argument("--all-ph", action="store_true", help="Run at pH 7, 5, 3.5, 2 (or --ph-values if given) and write results to per-pH subdirectories, plus a comparison plot")
    p.add_argument("--ph-values", default=None,
                    help="Comma-separated list of pH values for a multi-pH run (implies --all-ph), "
                    "e.g. '6.0,6.4,6.8,7.0,7.4,7.8' for a fine sweep over a receptor's physiological range")
    p.add_argument("--pka-override", default=None,
                    help="Comma-separated author-resnum:pKa pairs to override the default per-residue-type pKa, "
                    "e.g. '17:7.4,269:7.0' for candidate pH-sensor histidines with a shifted pKa")
    p.add_argument("--block-size", type=int, default=4, help="Residues per block (default: 4)")
    p.add_argument("--preset", choices=["membrane", "soluble"], default="membrane",
                    help="Parameter preset: membrane/GPCR (dielectric=4, default) or soluble protein (dielectric=29)")
    p.add_argument("--temp", type=float, default=None, help="Temperature in K (default: preset's, 310)")
    p.add_argument("--ene", type=float, default=None, help="vdW energy per native contact, kJ/mol (override preset)")
    p.add_argument("--ds", type=float, default=None, help="Entropic cost per residue, kJ/mol/K (override preset)")
    p.add_argument("--dcp", type=float, default=None, help="Heat capacity change per contact, kJ/mol/K (override preset)")
    p.add_argument("--ionic-strength", type=float, default=None, help="Ionic strength, M (override preset)")
    p.add_argument("--dielectric", type=float, default=None, help="Medium dielectric constant (override preset)")
    p.add_argument("--ss-codes", default=None, help="Explicit per-residue SS code string (H/E/G/other), overrides geometric assignment")
    p.add_argument("--ss-file", default=None, help="Path to a file whose contents are per-residue SS codes (H/E/G/other)")
    p.add_argument("--use-dssp", action="store_true",
                    help="Run real DSSP (requires mkdssp on PATH) instead of the geometric heuristic; "
                         "measurably closer to the original tool's real STRIDE-based blocking. "
                         "Ignored if --ss-codes/--ss-file is also given (explicit codes take priority).")
    p.add_argument("--out-dir", default="wsme_output", help="Output directory (default: wsme_output)")
    p.add_argument("--dsc", action="store_true", help="Also compute a DSC thermogram (temperature sweep; slower)")
    p.add_argument("--dsc-tmin", type=float, default=273.0)
    p.add_argument("--dsc-tmax", type=float, default=373.0)
    p.add_argument("--dsc-tstep", type=float, default=1.0)
    p.add_argument("--coupling", action="store_true",
                    help="Also compute the residue-residue coupling free-energy matrix "
                    "(thermodynamic coupling between block pairs; comparable cost to the landscape itself)")
    p.add_argument("--alanine-scan", action="store_true",
                    help="Also run in silico alanine-scanning mutagenesis (Fig. 7 of Anantakrishnan & Naganathan, "
                    "Nat Commun 2023): mutate each scanned residue to Ala, recompute the positive coupling free "
                    "energy matrix, and compare to wild type. Applies to ANY structure, not receptor-specific. "
                    "Uses --ph (not --all-ph/--ph-values, which this ignores). Costs roughly one coupling-matrix "
                    "computation per scanned residue -- see the printed time estimate before it runs.")
    p.add_argument("--ala-positions", default=None,
                    help="Comma-separated author resnums to scan (default: every eligible residue, i.e. all "
                    "except existing Ala/Gly/Pro -- see --ala-max-n to cap this for a faster run)")
    p.add_argument("--ala-max-n", type=int, default=40,
                    help="Cap the scan to this many evenly-spaced positions across the sequence (default: 40; "
                    "pass 0 or use --ala-positions with an explicit list to scan everything / a specific set)")
    p.add_argument("--ala-top-n", type=int, default=5,
                    help="Number of top-perturbing mutations (by total |mean DeltaDeltaG+|) to generate "
                    "distance-dependence and structure-map plots for (default: 5)")
    p.add_argument("--ala-n-clusters", type=int, default=4,
                    help="Number of k-means clusters for the PCA plot of per-residue coupling "
                    "perturbation across pH (default: 4; only used with --ala-all-ph)")
    p.add_argument("--ala-all-ph", action="store_true",
                    help="Run the alanine scan independently at every pH in --all-ph/--ph-values instead of "
                    "once at --ph. Mutation effects on coupling are themselves pH-dependent (pH changes which "
                    "atoms are charged, which feeds the contact map each mutant is compared against), so this "
                    "is the complete answer to 'how does this mutation's effect change with pH' -- at the cost "
                    "of one full scan PER pH value. Requires --alanine-scan and --all-ph/--ph-values.")
    p.add_argument("--no-plots", action="store_true", help="Skip generating plot images")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ss_codes = None
    if args.ss_codes or args.ss_file:
        if args.use_dssp:
            p.error("--use-dssp cannot be combined with --ss-codes/--ss-file (explicit codes already "
                     "specify a source; pick one)")
        ss_codes = args.ss_codes if args.ss_codes else Path(args.ss_file).read_text().strip()

    pka_overrides = None
    if args.pka_override:
        pka_overrides = {}
        for pair in args.pka_override.split(","):
            resnum, pka = pair.split(":")
            pka_overrides[int(resnum)] = float(pka)

    params = _build_params(args)
    dsc_T_grid = None
    if args.dsc:
        dsc_T_grid = np.arange(args.dsc_tmin, args.dsc_tmax + args.dsc_tstep / 2, args.dsc_tstep)

    if args.all_ph or args.ph_values:
        ph_values = [float(v) for v in args.ph_values.split(",")] if args.ph_values else list(DEFAULT_PH_VALUES)
        print(f"Running at pH values: {ph_values}")

        def progress(ph, i, total):
            print(f"[{i + 1}/{total}] pH {ph} ...")

        results = run_pipeline_multi_ph(
            args.pdb, ph_values=ph_values, chain=args.chain, model=args.model, pka_overrides=pka_overrides,
            ss_codes=ss_codes, use_dssp=args.use_dssp, block_size=args.block_size, params=params,
            with_dsc=args.dsc, dsc_T_grid=dsc_T_grid, with_coupling=args.coupling, progress_callback=progress,
        )
        for ph, pr in results.items():
            _report(pr)
            ph_dir = out_dir / f"pH_{ph}"
            ph_dir.mkdir(parents=True, exist_ok=True)
            _write_outputs(ph_dir, pr, save_plot=not args.no_plots)
            print(f"  wrote outputs to {ph_dir}")

        if not args.no_plots:
            from .plotting import plot_1d_profile_comparison, plot_2d_landscape_surface_comparison, save_figure
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_1d_profile_comparison({ph: pr.result for ph, pr in results.items()}, ax=ax)
            ax.set_title("1D Free Energy Profile vs. pH")
            fig.tight_layout()
            written = save_figure(fig, out_dir / "pH_comparison.png")
            plt.close(fig)
            print(f"Wrote {written[0].name} / {written[1].name}")

            fig = plot_2d_landscape_surface_comparison({f"pH {ph}": pr.result for ph, pr in results.items()})
            fig.suptitle("3D Free Energy Landscape vs. pH", fontsize=14)
            written = save_figure(fig, out_dir / "pH_comparison_3D.png", bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {written[0].name} / {written[1].name}")

        if args.alanine_scan:
            if args.ala_all_ph:
                _run_alanine_scan_multi_ph_cli(args, out_dir, params, pka_overrides, ss_codes, ph_values)
            else:
                _run_alanine_scan_cli(args, out_dir, params, pka_overrides, ss_codes)
        return

    if args.ala_all_ph:
        p.error("--ala-all-ph requires --all-ph or --ph-values")

    pr = run_pipeline(
        args.pdb, chain=args.chain, model=args.model, ph=args.ph, pka_overrides=pka_overrides, ss_codes=ss_codes,
        use_dssp=args.use_dssp, block_size=args.block_size, params=params, with_dsc=args.dsc,
        dsc_T_grid=dsc_T_grid, with_coupling=args.coupling,
    )
    _report(pr)
    _write_outputs(out_dir, pr, save_plot=not args.no_plots)
    print(f"Wrote outputs to {out_dir}")

    if args.alanine_scan:
        _run_alanine_scan_cli(args, out_dir, params, pka_overrides, ss_codes)


def _run_alanine_scan_cli(args, out_dir: Path, params: WSMEParams, pka_overrides: dict, ss_codes: str):
    from .alanine_scan import estimate_scan_seconds, scannable_positions
    from .structure import load_structure

    positions = None
    if args.ala_positions:
        positions = [int(v) for v in args.ala_positions.split(",")]
    max_positions = args.ala_max_n if args.ala_max_n and args.ala_max_n > 0 else None

    n_estimate = len(positions) if positions else (
        max_positions if max_positions else len(scannable_positions(load_structure(args.pdb, chain=args.chain, model=args.model, ph=args.ph)))
    )
    est_seconds = estimate_scan_seconds(n_estimate)
    print(f"\nAlanine scan (at pH {args.ph}, the single --ph value -- mutational scanning is not "
          f"repeated across --all-ph/--ph-values): {n_estimate} position(s) + WT baseline, "
          f"estimated ~{est_seconds / 60:.1f} min "
          f"(timing scales with structure size; first mutant's time is a better estimate for very large structures)")

    def progress(resnum, i, total, elapsed):
        rate = elapsed / (i + 1)
        remaining = rate * (total - i - 1)
        print(f"  [{i + 1}/{total}] resnum {resnum} done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    scan_pr = run_alanine_scan_pipeline(
        args.pdb, chain=args.chain, model=args.model, ph=args.ph, pka_overrides=pka_overrides, ss_codes=ss_codes,
        use_dssp=args.use_dssp, block_size=args.block_size, params=params, positions=positions,
        max_positions=max_positions, progress_callback=progress,
    )
    ala_dir = out_dir / "alanine_scan"
    ala_dir.mkdir(parents=True, exist_ok=True)
    _write_alanine_scan_outputs(ala_dir, scan_pr, top_n=args.ala_top_n, save_plot=not args.no_plots)
    print(f"Alanine scan: wrote outputs to {ala_dir}")


def _run_alanine_scan_multi_ph_cli(args, out_dir: Path, params: WSMEParams, pka_overrides: dict, ss_codes: str, ph_values: list):
    from .alanine_scan import estimate_scan_seconds, ph_sensitivity_table, scannable_positions
    from .pipeline import run_alanine_scan_pipeline_multi_ph
    from .structure import load_structure

    positions = None
    if args.ala_positions:
        positions = [int(v) for v in args.ala_positions.split(",")]
    max_positions = args.ala_max_n if args.ala_max_n and args.ala_max_n > 0 else None

    n_estimate = len(positions) if positions else (
        max_positions if max_positions else len(scannable_positions(load_structure(args.pdb, chain=args.chain, model=args.model, ph=ph_values[0])))
    )
    est_min_per_ph = estimate_scan_seconds(n_estimate) / 60
    print(f"\nAlanine scan across {len(ph_values)} pH values {ph_values}: {n_estimate} position(s) + WT "
          f"baseline PER pH, estimated ~{est_min_per_ph:.1f} min/pH, ~{est_min_per_ph * len(ph_values):.1f} min "
          f"total. This is a long-running, receptor-wide analysis run once per pH -- mutation effects on "
          f"coupling are themselves pH-dependent, so this is the complete (not approximate) answer.")

    def progress(ph, ph_i, ph_total, resnum, i, total, elapsed):
        rate = elapsed / (i + 1)
        remaining = rate * (total - i - 1)
        print(f"  [pH {ph} {ph_i + 1}/{ph_total}] [{i + 1}/{total}] resnum {resnum} done "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining this pH)")

    results = run_alanine_scan_pipeline_multi_ph(
        args.pdb, ph_values=ph_values, chain=args.chain, model=args.model, pka_overrides=pka_overrides,
        ss_codes=ss_codes, use_dssp=args.use_dssp, block_size=args.block_size, params=params,
        positions=positions, max_positions=max_positions, progress_callback=progress,
    )

    ala_dir = out_dir / "alanine_scan"
    ala_dir.mkdir(parents=True, exist_ok=True)
    for ph, scan_pr in results.items():
        ph_dir = ala_dir / f"pH_{ph}"
        ph_dir.mkdir(parents=True, exist_ok=True)
        _write_alanine_scan_outputs(ph_dir, scan_pr, top_n=args.ala_top_n, save_plot=not args.no_plots)
        print(f"  pH {ph}: wrote outputs to {ph_dir}")

    scan_by_ph = {ph: pr.scan for ph, pr in results.items()}
    sens_table = ph_sensitivity_table(scan_by_ph, n=args.ala_top_n)
    ph_cols = sorted(results)
    with open(ala_dir / "pH_Sensitivity.txt", "w") as f:
        f.write("# Mutation sites ranked by how much their perturbation magnitude\n")
        f.write("# (sum |<DeltaDeltaG+>|) swings across the pH values scanned -- a large\n")
        f.write("# swing flags a mutation whose apparent coupling role looks pH-modulated,\n")
        f.write("# a candidate conformational pH sensor.\n")
        f.write("# " + "  ".join(["resnum"] + [f"score_pH{ph}" for ph in ph_cols] + ["ph_spread"]) + "\n")
        for row in sens_table:
            vals = [f"{row['scores_by_ph'][ph]:10.3f}" for ph in ph_cols]
            f.write(f"{row['resnum']:6d}  " + "  ".join(vals) + f"  {row['ph_spread']:10.3f}\n")
    print(f"  pH sensitivity (top swings): "
          f"{[(r['resnum'], round(r['ph_spread'], 2)) for r in sens_table[:5]]}")

    from .alanine_scan import ph_cluster_table

    cluster_rows = ph_cluster_table(scan_by_ph, n_clusters=args.ala_n_clusters)
    with open(ala_dir / "PCA_Cluster.csv", "w") as f:
        f.write("resnum,cluster,magnitude,ph_spread,pc1,pc2\n")
        for row in cluster_rows:
            f.write(f"{row['resnum']},{row['cluster']},{row['magnitude']:.4f},"
                     f"{row['ph_spread']:.4f},{row['pc1']:.4f},{row['pc2']:.4f}\n")
    print(f"  Wrote {ala_dir / 'PCA_Cluster.csv'} ({len(cluster_rows)} residues, "
          f"{args.ala_n_clusters} clusters)")

    if not args.no_plots:
        from .plotting import (
            plot_alanine_ph_magnitude_vs_sensitivity,
            plot_alanine_ph_pca,
            plot_mutational_response_comparison,
            save_figure,
        )
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_mutational_response_comparison(scan_by_ph, ax=ax)
        ax.set_title("Mutational Response vs. pH")
        fig.tight_layout()
        written = save_figure(fig, ala_dir / "MutationalResponse_vs_pH.png")
        plt.close(fig)
        print(f"  Wrote {written[0].name} / {written[1].name}")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plot_alanine_ph_pca(scan_by_ph, ax=axes[0], n_clusters=args.ala_n_clusters, top_n_labels=args.ala_top_n)
        plot_alanine_ph_magnitude_vs_sensitivity(scan_by_ph, ax=axes[1], n_clusters=args.ala_n_clusters, top_n_labels=args.ala_top_n)
        fig.tight_layout()
        written = save_figure(fig, ala_dir / "PCA_Cluster.png")
        plt.close(fig)
        print(f"  Wrote {written[0].name} / {written[1].name}")

    print(f"Alanine scan (multi-pH): wrote outputs to {ala_dir}")


def _write_alanine_scan_outputs(out_dir: Path, scan_pr: AlanineScanPipelineResult, top_n: int, save_plot: bool):
    scan = scan_pr.scan
    with open(out_dir / "MutationalResponse.txt", "w") as f:
        f.write("# column 1: Block Index\n")
        f.write("# column 2: Mean mutational response <DeltaDeltaG+> across all scanned positions (kJ/mol)\n")
        f.write("# column 3: Std of mutational response (kJ/mol)\n")
        for b, (m, s) in enumerate(zip(scan.MR_mean, scan.MR_std)):
            f.write(f"{b:3d} {m:8.3f} {s:8.3f}\n")

    with open(out_dir / "DeltaDeltaG.csv", "w") as f:
        f.write("mutated_resnum,block,mean_ddG_plus\n")
        for row in scan.to_records():
            f.write(f"{row['mutated_resnum']},{row['block']},{row['mean_ddG+']:.4f}\n")

    top_hits = scan.top_hits(top_n)
    with open(out_dir / "TopHits.txt", "w") as f:
        f.write("# column 1: Mutated author resnum\n")
        f.write("# column 2: Total perturbation magnitude, sum(|<DeltaDeltaG+>|) (kJ/mol)\n")
        for resnum, score in top_hits:
            f.write(f"{resnum:6d} {score:10.3f}\n")
    print(f"  top {len(top_hits)} hits: {top_hits}")

    if save_plot:
        from .plotting import plot_ddg_structure_map, plot_ddg_vs_distance, plot_mutational_response, save_figure
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_mutational_response(scan, ax=ax, highlight={r: str(r) for r, _ in top_hits})
        fig.tight_layout()
        save_figure(fig, out_dir / "MutationalResponse.png")
        plt.close(fig)

        for resnum, _ in top_hits:
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_ddg_vs_distance(scan, resnum, ax=ax)
            fig.tight_layout()
            save_figure(fig, out_dir / f"DistanceDependence_{resnum}.png")
            plt.close(fig)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(projection="3d")
            plot_ddg_structure_map(scan, resnum, ax=ax)
            fig.tight_layout()
            save_figure(fig, out_dir / f"StructureMap_{resnum}.png")
            plt.close(fig)


def _report(pr: PipelineResult):
    s = pr.structure
    print(f"pH {pr.ph}: {s.nres} residues (chain {s.chain_id})")
    for w in pr.warnings:
        print(f"  warning: {w}")
    print(f"  {int(pr.ss_mask.sum())}/{s.nres} residues structured; "
          f"{int(pr.contact_map.srcont.sum())} VdW contacts, {len(pr.contact_map.elec_pairs)} electrostatic pairs; "
          f"{pr.block_model.nblocks} blocks")
    r = pr.result
    print(f"  Zfin={r.zfin:.4e}  states: SSA={r.stats['n_states_ssa']} DSA={r.stats['n_states_dsa']} DSAw/L={r.stats['n_states_dsawl']}")
    print(f"  partition fn %%: SSA={r.stats['pct_ssa']:.1f} DSA={r.stats['pct_dsa']:.1f} DSAw/L={r.stats['pct_dsawl']:.1f}")


def _write_outputs(out_dir: Path, pr: PipelineResult, save_plot: bool):
    result = pr.result
    _write_1d_profile(out_dir / "1D_FreeEnergyProfile.txt", result)
    _write_2d_surface(out_dir / "2D_FreeEnergySurface.txt", result)
    _write_fpath(out_dir / "ResFoldProb_vs_RC.txt", result)
    if pr.dsc_result is not None:
        _write_dsc(out_dir / "DSC_Thermogram.txt", pr.dsc_result)
    if pr.coupling_result is not None:
        _write_coupling(out_dir / "CouplingMatrix.txt", pr.coupling_result)
    if save_plot:
        from .plotting import plot_2d_landscape_surface, plot_coupling_matrix, plot_summary, save_figure
        import matplotlib.pyplot as plt

        fig = plot_summary(result, dsc_result=pr.dsc_result, coupling_result=pr.coupling_result, save_path=str(out_dir / "summary.png"))
        plt.close(fig)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        plot_2d_landscape_surface(result, ax=ax)
        fig.tight_layout()
        save_figure(fig, out_dir / "2D_FreeEnergyLandscape_3D.png")
        plt.close(fig)

        if pr.coupling_result is not None:
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_coupling_matrix(pr.coupling_result, ax=ax)
            fig.tight_layout()
            save_figure(fig, out_dir / "CouplingMatrix.png")
            plt.close(fig)


def _write_1d_profile(path, result):
    with open(path, "w") as f:
        f.write("# column 1: No. of Structured Blocks\n")
        f.write("# column 2: Free Energy (kJ/mol)\n")
        for n, fe in zip(result.n_values, result.fes):
            f.write(f"{n:3d} {fe:8.3f}\n")


def _write_2d_surface(path, result):
    with open(path, "w") as f:
        f.write("# column 1: No. of Structured Blocks in N-terminal half\n")
        f.write("# column 2: No. of Structured Blocks in C-terminal half\n")
        f.write("# column 3: Free Energy (kJ/mol)\n")
        for i in range(result.fes2D.shape[0]):
            for j in range(result.fes2D.shape[1]):
                f.write(f"{i:3d} {j:3d} {result.fes2D[i, j]:8.3f}\n")


def _write_fpath(path, result):
    with open(path, "w") as f:
        f.write("# column 1: No. of Structured Blocks\n")
        f.write("# column 2: Block Index\n")
        f.write("# column 3: Folding Probability\n")
        for ni, n in enumerate(result.n_values):
            for b in range(result.fpath.shape[1]):
                f.write(f"{n:3d} {b:3d} {result.fpath[ni, b]:1.3f}\n")


def _write_dsc(path, dsc_result):
    with open(path, "w") as f:
        f.write("# column 1: Temperature (K)\n")
        f.write("# column 2: Cp total (kJ/mol/K)\n")
        f.write("# column 3: Cp excess (from partition function) (kJ/mol/K)\n")
        for T, cp, cpx in zip(dsc_result.T, dsc_result.Cp, dsc_result.Cp_excess):
            f.write(f"{T:6.1f} {cp:10.5f} {cpx:10.5f}\n")


def _write_coupling(path, coupling_result):
    mat = coupling_result.coupling_free_energy
    nb = mat.shape[0]
    with open(path, "w") as f:
        f.write("# column 1: Block j\n")
        f.write("# column 2: Block k\n")
        f.write("# column 3: Coupling Free Energy (kJ/mol); positive = j,k tend to fold together\n")
        f.write("# column 4: P(j folded, k folded)\n")
        for j in range(nb):
            for k in range(nb):
                f.write(f"{j:3d} {k:3d} {mat[j, k]:8.3f} {coupling_result.p_folded_folded[j, k]:1.4f}\n")


if __name__ == "__main__":
    main()
