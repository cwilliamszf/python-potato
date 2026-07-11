"""Command-line interface: compute a bWSME conformational landscape for a
PDB structure and write profile/landscape/probability files plus plots,
mirroring the outputs of FesCalc_Block.m / DSCcalc_Block.m."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from .blocking import build_blocks
from .contacts import compute_contact_map
from .dsc import compute_dsc
from .secondary_structure import assign_secondary_structure, secondary_structure_from_codes
from .structure import load_structure
from .wsme import WSMEParams, run_wsme


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
    p.add_argument("--ph", type=float, default=7.0, choices=[7.0, 5.0, 3.5, 2.0], help="pH for charge assignment (default: 7.0)")
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

    print(f"Loading structure: {args.pdb}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = load_structure(args.pdb, chain=args.chain, model=args.model, ph=args.ph)
        for w in caught:
            print(f"  warning: {w.message}")
    print(f"  {structure.nres} residues, chain {structure.chain_id}")

    if args.ss_codes or args.ss_file:
        codes = args.ss_codes if args.ss_codes else Path(args.ss_file).read_text().strip()
        if len(codes) != structure.nres:
            sys.exit(f"SS code string length ({len(codes)}) != number of residues ({structure.nres})")
        ss_mask = secondary_structure_from_codes(codes)
        print("Using supplied secondary-structure codes")
    else:
        ss_mask = assign_secondary_structure(structure)
        print(f"Geometric secondary structure assignment: {ss_mask.sum()}/{structure.nres} residues structured")

    contact_map = compute_contact_map(structure)
    print(f"  {int(contact_map.srcont.sum())} VdW contacts, {len(contact_map.elec_pairs)} electrostatic pairs")

    block_model = build_blocks(ss_mask, contact_map, block_size=args.block_size)
    print(f"  {block_model.nblocks} blocks (block_size={args.block_size})")

    params = _build_params(args)
    print(f"Running WSME (T={params.T} K, preset={args.preset})...")
    result = run_wsme(structure, block_model, ss_mask, params)
    print(f"  Zfin={result.zfin:.4e}")
    print(f"  states: SSA={result.stats['n_states_ssa']} DSA={result.stats['n_states_dsa']} DSAw/L={result.stats['n_states_dsawl']}")
    print(f"  partition fn %%: SSA={result.stats['pct_ssa']:.1f} DSA={result.stats['pct_dsa']:.1f} DSAw/L={result.stats['pct_dsawl']:.1f}")

    _write_1d_profile(out_dir / "1D_FreeEnergyProfile.txt", result, params)
    _write_2d_surface(out_dir / "2D_FreeEnergySurface.txt", result, params)
    _write_fpath(out_dir / "ResFoldProb_vs_RC.txt", result, params)
    print(f"Wrote profile/landscape/probability files to {out_dir}")

    dsc_result = None
    if args.dsc:
        print(f"Running DSC sweep ({args.dsc_tmin}-{args.dsc_tmax} K, step {args.dsc_tstep})...")
        T_grid = np.arange(args.dsc_tmin, args.dsc_tmax + args.dsc_tstep / 2, args.dsc_tstep)
        dsc_result = compute_dsc(structure, block_model, ss_mask, params, T_grid=T_grid)
        _write_dsc(out_dir / "DSC_Thermogram.txt", dsc_result)
        print(f"Wrote {out_dir / 'DSC_Thermogram.txt'}")

    if not args.no_plots:
        from .plotting import plot_summary
        fig = plot_summary(result, dsc_result=dsc_result, save_path=str(out_dir / "summary.png"))
        print(f"Wrote {out_dir / 'summary.png'}")
        import matplotlib.pyplot as plt
        plt.close(fig)


def _write_1d_profile(path, result, params):
    with open(path, "w") as f:
        f.write("# column 1: No. of Structured Blocks\n")
        f.write("# column 2: Free Energy (kJ/mol)\n")
        for n, fe in zip(result.n_values, result.fes):
            f.write(f"{n:3d} {fe:8.3f}\n")


def _write_2d_surface(path, result, params):
    with open(path, "w") as f:
        f.write("# column 1: No. of Structured Blocks in N-terminal half\n")
        f.write("# column 2: No. of Structured Blocks in C-terminal half\n")
        f.write("# column 3: Free Energy (kJ/mol)\n")
        for i in range(result.fes2D.shape[0]):
            for j in range(result.fes2D.shape[1]):
                f.write(f"{i:3d} {j:3d} {result.fes2D[i, j]:8.3f}\n")


def _write_fpath(path, result, params):
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
