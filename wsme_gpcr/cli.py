"""Command-line interface: compute a bWSME conformational landscape for a
PDB structure and write profile/landscape/probability files plus plots,
mirroring the outputs of FesCalc_Block.m / DSCcalc_Block.m."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pipeline import DEFAULT_PH_VALUES, PipelineResult, run_pipeline, run_pipeline_multi_ph
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
    p.add_argument("--ph", type=float, default=7.0, choices=[7.0, 5.0, 3.5, 2.0], help="pH for charge assignment (default: 7.0; ignored with --all-ph)")
    p.add_argument("--all-ph", action="store_true", help="Run at every pH (7, 5, 3.5, 2) and write results to per-pH subdirectories, plus a comparison plot")
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
    p.add_argument("--out-dir", default="wsme_output", help="Output directory (default: wsme_output)")
    p.add_argument("--dsc", action="store_true", help="Also compute a DSC thermogram (temperature sweep; slower)")
    p.add_argument("--dsc-tmin", type=float, default=273.0)
    p.add_argument("--dsc-tmax", type=float, default=373.0)
    p.add_argument("--dsc-tstep", type=float, default=1.0)
    p.add_argument("--no-plots", action="store_true", help="Skip generating plot images")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ss_codes = None
    if args.ss_codes or args.ss_file:
        ss_codes = args.ss_codes if args.ss_codes else Path(args.ss_file).read_text().strip()

    params = _build_params(args)
    dsc_T_grid = None
    if args.dsc:
        dsc_T_grid = np.arange(args.dsc_tmin, args.dsc_tmax + args.dsc_tstep / 2, args.dsc_tstep)

    if args.all_ph:
        print(f"Running at all pH values: {DEFAULT_PH_VALUES}")

        def progress(ph, i, total):
            print(f"[{i + 1}/{total}] pH {ph} ...")

        results = run_pipeline_multi_ph(
            args.pdb, chain=args.chain, model=args.model, ss_codes=ss_codes,
            block_size=args.block_size, params=params, with_dsc=args.dsc,
            dsc_T_grid=dsc_T_grid, progress_callback=progress,
        )
        for ph, pr in results.items():
            _report(pr)
            ph_dir = out_dir / f"pH_{ph}"
            ph_dir.mkdir(parents=True, exist_ok=True)
            _write_outputs(ph_dir, pr, save_plot=not args.no_plots)
            print(f"  wrote outputs to {ph_dir}")

        if not args.no_plots:
            from .plotting import plot_1d_profile_comparison, plot_2d_landscape_surface_comparison
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            plot_1d_profile_comparison({ph: pr.result for ph, pr in results.items()}, ax=ax)
            ax.set_title("1D Free Energy Profile vs. pH")
            fig.tight_layout()
            fig.savefig(out_dir / "pH_comparison.png", dpi=200)
            plt.close(fig)
            print(f"Wrote {out_dir / 'pH_comparison.png'}")

            fig = plot_2d_landscape_surface_comparison({f"pH {ph}": pr.result for ph, pr in results.items()})
            fig.suptitle("3D Free Energy Landscape vs. pH", fontsize=14)
            fig.savefig(out_dir / "pH_comparison_3D.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out_dir / 'pH_comparison_3D.png'}")
        return

    pr = run_pipeline(
        args.pdb, chain=args.chain, model=args.model, ph=args.ph, ss_codes=ss_codes,
        block_size=args.block_size, params=params, with_dsc=args.dsc, dsc_T_grid=dsc_T_grid,
    )
    _report(pr)
    _write_outputs(out_dir, pr, save_plot=not args.no_plots)
    print(f"Wrote outputs to {out_dir}")


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
    if save_plot:
        from .plotting import plot_2d_landscape_surface, plot_summary
        import matplotlib.pyplot as plt

        fig = plot_summary(result, dsc_result=pr.dsc_result, save_path=str(out_dir / "summary.png"))
        plt.close(fig)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
        plot_2d_landscape_surface(result, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "2D_FreeEnergyLandscape_3D.png", dpi=200)
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


if __name__ == "__main__":
    main()
