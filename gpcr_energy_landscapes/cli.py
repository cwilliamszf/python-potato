"""
Command-line entry point.

    python -m gpcr_energy_landscapes landscape1d \
        --ensemble conformers/ --energies gibbs.csv \
        --cv-set beta2ar --cv tm5_bulge --out tm5_bulge_landscape.png

    python -m gpcr_energy_landscapes landscape2d \
        --ensemble conformers/ --energies gibbs.csv \
        --cv-set beta2ar --cv-x tm5_bulge --cv-y ionic_lock --out landscape2d.png

    python -m gpcr_energy_landscapes embed \
        --ensemble conformers/ --energies gibbs.csv \
        --cv-set beta2ar --method pca --out embedding.png
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from . import io, pipeline, plotting
from .collective_variables import BETA2AR_MICROSWITCHES

_CV_SETS = {"beta2ar": BETA2AR_MICROSWITCHES}


def _load_refs(args) -> dict | None:
    if not args.active_ref or not args.inactive_ref:
        return None
    return {"active": io.load_structure(args.active_ref), "inactive": io.load_structure(args.inactive_ref)}


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ensemble", required=True, help="Directory of conformer PDB files (tool 2 output)")
    parser.add_argument("--energies", required=True, help="CSV of per-structure Gibbs free energies (tool 3 output)")
    parser.add_argument("--id-col", default="structure_id")
    parser.add_argument("--gibbs-col", default="gibbs_kcal_mol")
    parser.add_argument("--weight-col", default=None)
    parser.add_argument("--cv-set", default="beta2ar", choices=sorted(_CV_SETS))
    parser.add_argument("--active-ref", default=None, help="Active-state reference PDB (for connector_delta_rmsd CVs)")
    parser.add_argument("--inactive-ref", default=None, help="Inactive-state reference PDB (for connector_delta_rmsd CVs)")
    parser.add_argument("--temperature", type=float, default=310.0)
    parser.add_argument("--method", default="kde", choices=["kde", "histogram"])
    parser.add_argument("--out", required=True, help="Output image path")


def _prepare(args):
    ensemble = io.load_ensemble(args.ensemble)
    energies = io.load_energies(args.energies, id_col=args.id_col, gibbs_col=args.gibbs_col)
    refs = _load_refs(args)
    cv_defs = _CV_SETS[args.cv_set]
    cv_table = pipeline.compute_cv_table(ensemble, cv_defs, refs=refs)
    return pipeline.merge_with_energies(cv_table, energies)


def _cmd_landscape1d(args) -> None:
    merged = _prepare(args)
    landscape = pipeline.build_1d_landscape(
        merged, args.cv, gibbs_col=args.gibbs_col, weight_col=args.weight_col, temperature=args.temperature, method=args.method
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    plotting.plot_1d_landscape(landscape, cv_label=args.cv, ax=ax)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


def _cmd_landscape2d(args) -> None:
    merged = _prepare(args)
    landscape = pipeline.build_2d_landscape(
        merged,
        args.cv_x,
        args.cv_y,
        gibbs_col=args.gibbs_col,
        weight_col=args.weight_col,
        temperature=args.temperature,
        method=args.method,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plotting.plot_2d_landscape(
        landscape, x_label=args.cv_x, y_label=args.cv_y, ax=ax, scatter=merged, scatter_x=args.cv_x, scatter_y=args.cv_y
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


def _cmd_embed(args) -> None:
    merged = _prepare(args)
    cv_defs = _CV_SETS[args.cv_set]
    feature_cols = [c["name"] for c in cv_defs if c["type"] != "connector_delta_rmsd"]
    embedding_df, _ = pipeline.build_embedding_landscape(
        merged, feature_cols=feature_cols, method=args.embed_method, gibbs_col=args.gibbs_col
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    cols = list(embedding_df.columns)
    plotting.plot_embedding(embedding_df, cols[0], cols[1], color_col=args.gibbs_col, ax=ax)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gpcr-landscape", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser("landscape1d", help="1D free-energy landscape along one CV")
    _common_args(p1)
    p1.add_argument("--cv", required=True)
    p1.set_defaults(func=_cmd_landscape1d)

    p2 = subparsers.add_parser("landscape2d", help="2D free-energy landscape along two CVs")
    _common_args(p2)
    p2.add_argument("--cv-x", required=True)
    p2.add_argument("--cv-y", required=True)
    p2.set_defaults(func=_cmd_landscape2d)

    p3 = subparsers.add_parser("embed", help="PCA/MDS/t-SNE embedding colored by free energy")
    _common_args(p3)
    p3.add_argument("--embed-method", default="pca", choices=["pca", "mds", "tsne"])
    p3.set_defaults(func=_cmd_embed)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
