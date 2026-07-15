"""
End-to-end demo of tool 4 using entirely synthetic data, so it runs without
needing real output from tools 1-3.

It fabricates a ~120-conformer "ensemble" spread across two basins (an
active-like basin and an inactive-like basin, plus a sparser bridge between
them) along four synthetic microswitch distances, assigns each conformer a
Gibbs free energy drawn so the active-like basin is lower energy (mimicking
an agonist-bound receptor), and then runs the full pipeline: collective
variables -> merge with energies -> 1D landscape, 2D landscape, and a PCA
embedding colored by free energy -- reproducing the shapes of Figure 2 and
Figure 3 of Fleetwood et al. 2021 (eLife 2021;10:e60715).

Run with:  python examples/synthetic_demo.py
Writes:    examples/output/{landscape_1d,landscape_2d,pca_embedding}.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO

from gpcr_energy_landscapes import pipeline, plotting
from gpcr_energy_landscapes.collective_variables import BETA2AR_MICROSWITCHES

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from helpers import make_structure  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "output"
N_PER_BASIN = 50
N_BRIDGE = 20
RNG = np.random.default_rng(42)


def _fabricate_ensemble():
    """Two basins ~ what a diverse AlphaFold-derived ensemble of an agonist-
    bound GPCR might look like: many active-like conformers, fewer
    inactive-like ones, and a handful bridging the two along the activation
    pathway."""

    def sample_basin(center, spread, n):
        return {
            "tm5_bulge": RNG.normal(center[0], spread, n),
            "ionic_lock": RNG.normal(center[1], spread * 1.3, n),
            "y_y_motif": RNG.normal(center[2], spread * 0.8, n),
            "connector_shift": RNG.normal(center[3], spread * 1.1, n),
        }

    active = sample_basin(center=(1.1, 1.0, 6.5, 0.0), spread=0.12, n=N_PER_BASIN)
    inactive = sample_basin(center=(1.4, 1.6, 7.2, 5.0), spread=0.15, n=int(N_PER_BASIN * 0.6))
    bridge = sample_basin(center=(1.25, 1.3, 6.85, 2.5), spread=0.25, n=N_BRIDGE)

    ensemble = {}
    gibbs = {}
    for label, basin, base_energy in (("active", active, -3.0), ("inactive", inactive, 0.0), ("bridge", bridge, 1.5)):
        n = len(basin["tm5_bulge"])
        for i in range(n):
            sid = f"{label}_{i:03d}"
            residues = {
                207: {"CA": (0.0, 0.0, 0.0)},
                315: {"CA": (0.0, 0.0, float(basin["tm5_bulge"][i]))},
                268: {"CA": (10.0, 0.0, 0.0)},
                131: {"CA": (10.0, 0.0, float(basin["ionic_lock"][i]))},
                219: {"CZ": (20.0, 0.0, 0.0)},
                326: {"CZ": (20.0, 0.0, float(basin["y_y_motif"][i]))},
                121: {"CA": (0.0, 0.0, float(basin["connector_shift"][i]))},
                282: {"CA": (1.0, 0.0, float(basin["connector_shift"][i]))},
            }
            ensemble[sid] = make_structure(sid, "A", residues)
            gibbs[sid] = base_energy + RNG.normal(0, 0.6)

    energies = pd.DataFrame({"structure_id": list(gibbs.keys()), "gibbs_kcal_mol": list(gibbs.values())})
    return ensemble, energies.set_index("structure_id")


def _write_example_pdbs(ensemble, out_dir, n=3):
    """Write a few example conformers to disk, illustrating the file layout
    io.load_ensemble() expects from tool 2's real output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    for sid in list(ensemble)[:n]:
        writer.set_structure(ensemble[sid])
        writer.save(str(out_dir / f"{sid}.pdb"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ensemble, energies = _fabricate_ensemble()
    refs = {
        "active": make_structure("active_ref", "A", {121: {"CA": (0, 0, 0)}, 282: {"CA": (1, 0, 0)}}),
        "inactive": make_structure("inactive_ref", "A", {121: {"CA": (0, 0, 5)}, 282: {"CA": (1, 0, 5)}}),
    }

    cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
    merged = pipeline.merge_with_energies(cv_table, energies)
    print(f"Ensemble: {len(merged)} conformers, CVs: {list(cv_table.columns)}")

    # --- Figure 2a-style 1D landscape along one microswitch ---
    landscape1d = pipeline.build_1d_landscape(merged, "connector_drmsd", method="kde")
    fig, ax = plt.subplots(figsize=(5, 4))
    plotting.plot_1d_landscape(landscape1d, cv_label="Connector ΔRMSD (Å)", ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "landscape_1d.png", dpi=200)
    plt.close(fig)

    # --- Figure 2b-style 2D landscape along two microswitches ---
    landscape2d = pipeline.build_2d_landscape(merged, "tm5_bulge", "ionic_lock", method="kde")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plotting.plot_2d_landscape(
        landscape2d,
        x_label="TM5 bulge (Å)",
        y_label="Ionic lock distance (Å)",
        ax=ax,
        scatter=merged,
        scatter_x="tm5_bulge",
        scatter_y="ionic_lock",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "landscape_2d.png", dpi=200)
    plt.close(fig)

    # --- Figure 3-style PCA embedding colored by free energy ---
    embedding_df, _ = pipeline.build_embedding_landscape(
        merged, feature_cols=["tm5_bulge", "ionic_lock", "y_y_motif"], method="pca"
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    plotting.plot_embedding(embedding_df, "pca_1", "pca_2", color_col="gibbs_kcal_mol", ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_embedding.png", dpi=200)
    plt.close(fig)

    _write_example_pdbs(ensemble, OUT_DIR / "example_conformers")
    energies.to_csv(OUT_DIR / "example_energies.csv")

    print(f"Wrote landscape_1d.png, landscape_2d.png, pca_embedding.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
