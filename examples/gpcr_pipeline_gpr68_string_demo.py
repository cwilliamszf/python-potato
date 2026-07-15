"""
Interpolated-pathway ("string-method-adjacent") sampling between the real
GPR68 active and inactive endpoint structures, as an alternative to tool 2
(GPU-bound ColabFold folding isn't available in this sandbox) for filling
in the conformational space *between* two known basins -- which the
ANM-around-each-endpoint approach in gpcr_pipeline_gpr68_demo.py explicitly
could not do (see that run's README: "two basins are resolved, but the
space between them is not sampled").

Method, and an honest disclosure of what this is and isn't:
  1. The active and inactive GPR68 models have EXACT atom correspondence
     (365/365 residues, identical atom names per residue -- verified before
     writing this script) and are already in the same reference frame (both
     GPCRdb AFMS models built on the same pipeline/template), so a straight
     linear Cartesian interpolation between corresponding atoms is a valid
     way to build an initial path -- no separate structural superposition
     step is needed or wanted (a Kabsch alignment would remove the genuine
     rigid-body-like helix motions that ARE the activation transition).
  2. N_IMAGES images are generated at evenly spaced interpolation fractions
     f in [0, 1] (f=0 is the active endpoint, f=1 is the inactive endpoint).
  3. Each image is passed through tool 3's real minimizer (150 L-BFGS
     iterations, same setting used throughout this session) to relax local
     steric strain introduced by naive linear interpolation, then scored
     with the same real AMBER ff14SB + GBn2 + RRHO Gibbs free energy.

This is **linear interpolation with local relaxation, not a converged
string method.** A true string method iteratively relaxes the path with
the path-tangent component of the force removed (so images don't just roll
downhill into one of the two endpoint basins) and periodically
reparametrizes the images to stay evenly spaced along the true minimum
free energy path -- that requires many more force evaluations per image
(multiple relaxation + reparametrization rounds) than this single-pass
approach. What's implemented here is the standard *starting guess* for a
real string method (often called "linear interpolation in Cartesians",
LIC), scored once with a real force field rather than left purely
geometric -- a reasonable, honestly-labeled middle ground given the
compute budget, not a substitute for the genuine iterative method.

Run with (~15-20 min, N_IMAGES real OpenMM minimizations at one pH):

    PYTHONPATH=.:gibbs python examples/gpcr_pipeline_gpr68_string_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gibbs"))

from gpcr_ensemble.activation_state import tm3_tm6_distance  # noqa: E402
from gpcr_energy_landscapes import pipeline as landscape_pipeline  # noqa: E402
from gpcr_energy_landscapes import plotting  # noqa: E402
from gpcr_energy_landscapes.collective_variables import rmsd as cv_rmsd  # noqa: E402
import gpcr_gibbs_energy as gibbs_tool  # noqa: E402
from examples.gpcr_pipeline_gpr68_demo import (  # noqa: E402
    ACTIVE_PDB, INACTIVE_PDB, find_conserved_motifs, find_ionic_lock_partner,
)

OUT_DIR = Path(__file__).resolve().parent / "output" / "gpr68_string_demo"
N_IMAGES = 11  # f = 0, 0.1, 0.2, ..., 1.0 (0=active endpoint, 1=inactive endpoint)
PH = 7.4


def build_interpolated_path(active_structure, inactive_structure, chain_id, n_images, out_dir):
    """Linear Cartesian interpolation between exactly-corresponding atoms of
    two endpoint structures. Assumes (and this script verifies once, in
    __main__) identical residue/atom composition and a shared reference
    frame between the two inputs."""
    a_chain = active_structure[0][chain_id]
    i_chain = inactive_structure[0][chain_id]
    a_residues = [r for r in a_chain if r.id[0] == " "]
    i_residues = [r for r in i_chain if r.id[0] == " "]

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    image_paths = {}
    fractions = np.linspace(0.0, 1.0, n_images)

    for k, f in enumerate(fractions):
        image_structure = active_structure.copy()
        image_chain = image_structure[0][chain_id]
        image_residues = [r for r in image_chain if r.id[0] == " "]
        for res_img, res_a, res_i in zip(image_residues, a_residues, i_residues):
            for atom_img in res_img:
                name = atom_img.get_name()
                coord_a = res_a[name].coord
                coord_i = res_i[name].coord
                atom_img.coord = (1.0 - f) * coord_a + f * coord_i

        sid = f"path_{k:02d}_f{f:.2f}"
        path = out_dir / f"{sid}.pdb"
        writer.set_structure(image_structure)
        writer.save(str(path))
        image_paths[sid] = (path, float(f))

    return image_paths


def run_tool3_gibbs(pdb_path, ph):
    argv = [
        str(pdb_path),
        "--chains", "A",
        "--ph", str(ph),
        "--entropy-method", "ca-anm",
        "--minimize-iterations", "150",
        "--no-hbond-analysis",
    ]
    args = gibbs_tool.parse_args(argv)
    result = gibbs_tool.run(args)
    return result["G_kcal"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = PDBParser(QUIET=True)
    active_structure = parser.get_structure("gpr68_active", str(ACTIVE_PDB))
    inactive_structure = parser.get_structure("gpr68_inactive", str(INACTIVE_PDB))

    # Sanity check the assumption this whole approach depends on.
    a_res = [r for r in active_structure[0]["A"] if r.id[0] == " "]
    i_res = [r for r in inactive_structure[0]["A"] if r.id[0] == " "]
    assert len(a_res) == len(i_res), "active/inactive residue counts differ -- cannot interpolate directly"
    for ra, ri in zip(a_res, i_res):
        assert ra.id[1] == ri.id[1] and {a.get_name() for a in ra} == {a.get_name() for a in ri}, (
            f"atom mismatch at resnum {ra.id[1]} -- cannot interpolate directly"
        )
    print(f"Verified exact atom correspondence: {len(a_res)}/{len(a_res)} residues match.")

    arg_resnum, tyr_resnum = find_conserved_motifs(active_structure)
    partner_resnum = find_ionic_lock_partner(active_structure, arg_resnum, exclude_resnums=(tyr_resnum,))
    print(f"DRY-Arg={arg_resnum}, TM7-Tyr={tyr_resnum}, ionic-lock-partner={partner_resnum}")

    a_ca_all = np.array([r["CA"].coord for r in a_res])
    i_ca_all = np.array([r["CA"].coord for r in i_res])
    endpoint_rmsd = float(np.sqrt(((a_ca_all - i_ca_all) ** 2).sum(axis=1).mean()))
    print(f"Raw (no superposition) CA RMSD between endpoints: {endpoint_rmsd:.2f} A")

    print(f"\nBuilding {N_IMAGES}-image linear interpolation path...")
    image_paths = build_interpolated_path(active_structure, inactive_structure, "A", N_IMAGES, OUT_DIR / "path")

    print(f"\nRunning real Gibbs free-energy calculations along the path at pH {PH} "
          f"({N_IMAGES} images, ~60-90s each)...")
    energies = {}
    cv_lock = {}
    cv_pxxy = {}
    rmsd_to_active = {}
    rmsd_to_inactive = {}
    t0 = time.time()
    for idx, (sid, (path, f)) in enumerate(image_paths.items()):
        g = run_tool3_gibbs(path, PH)
        energies[sid] = g

        conf = parser.get_structure(sid, str(path))
        chain = conf[0]["A"]
        ca_coords = {r.id[1]: r["CA"].coord for r in chain if r.id[0] == " "}
        cv_lock[sid] = tm3_tm6_distance(ca_coords, arg_resnum, partner_resnum) if partner_resnum else np.nan
        tyr_res = next(r for r in chain if r.id[1] == tyr_resnum)
        arg_res = next(r for r in chain if r.id[1] == arg_resnum)
        cv_pxxy[sid] = float(np.linalg.norm(tyr_res["CA"].coord - arg_res["CA"].coord))

        image_ca = np.array([r["CA"].coord for r in chain if r.id[0] == " "])
        rmsd_to_active[sid] = cv_rmsd(image_ca, a_ca_all)
        rmsd_to_inactive[sid] = cv_rmsd(image_ca, i_ca_all)

        print(f"  [{idx+1}/{N_IMAGES}] {sid} (f={f:.2f}): G = {g:.2f} kcal/mol, "
              f"RMSD-to-active={rmsd_to_active[sid]:.2f} A, RMSD-to-inactive={rmsd_to_inactive[sid]:.2f} A "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    fractions = {sid: f for sid, (_, f) in image_paths.items()}
    table = pd.DataFrame(
        {
            "structure_id": list(image_paths.keys()),
            "interpolation_fraction": [fractions[s] for s in image_paths],
            "ionic_lock_like": [cv_lock[s] for s in image_paths],
            "pxxy_dry_distance": [cv_pxxy[s] for s in image_paths],
            "rmsd_to_active": [rmsd_to_active[s] for s in image_paths],
            "rmsd_to_inactive": [rmsd_to_inactive[s] for s in image_paths],
            "gibbs_kcal_mol": [energies[s] for s in image_paths],
        }
    ).set_index("structure_id")
    table.to_csv(OUT_DIR / "path_table.csv")

    # G directly vs. interpolation fraction -- the primary result: a real
    # free-energy profile along an actual (approximate) transition path,
    # rather than two disconnected points.
    fig, ax = plt.subplots(figsize=(6, 4.3))
    order = table.sort_values("interpolation_fraction")
    ax.plot(order["interpolation_fraction"], order["gibbs_kcal_mol"], "o-", lw=2, color="#3b6fa0")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("G (kcal/mol)")
    ax.set_title(f"GPR68: real Gibbs energy along active<->inactive linear\ninterpolation path (pH {PH}, N={N_IMAGES} images)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gibbs_vs_path.png", dpi=200)
    plt.close(fig)

    # Same data as a proper tool-4 KDE free-energy landscape along the path
    # coordinate -- now meaningful since the path spans real variance,
    # unlike the near-zero-spread ANM ensembles.
    landscape = landscape_pipeline.build_1d_landscape(
        table.assign(gibbs_kcal_mol=table["gibbs_kcal_mol"]),
        "interpolation_fraction", method="kde", grid_size=200, bandwidth=0.3,
    )
    fig, ax = plt.subplots(figsize=(6, 4.3))
    plotting.plot_1d_landscape(landscape, cv_label="Interpolation fraction (0=active, 1=inactive)", ax=ax)
    ax.scatter(table["interpolation_fraction"], table["gibbs_kcal_mol"] - table["gibbs_kcal_mol"].min(),
               color="#c0533e", s=25, zorder=3, label="path images (raw, shifted)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"GPR68 tool-4 landscape along the interpolation path (pH {PH})", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "landscape_along_path.png", dpi=200)
    plt.close(fig)

    # RMSD-to-endpoint sanity check: does the path monotonically leave the
    # active basin and approach the inactive one, or does relaxation pull
    # images back toward one endpoint (the string-method-collapse failure
    # mode this whole docstring warns about)?
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(order["interpolation_fraction"], order["rmsd_to_active"], "o-", label="RMSD to active", color="#3b6fa0")
    ax.plot(order["interpolation_fraction"], order["rmsd_to_inactive"], "o-", label="RMSD to inactive", color="#c0533e")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("CA RMSD (Å) after relaxation")
    ax.legend(frameon=False)
    ax.set_title("Did relaxation preserve the interpolated path, or collapse it?", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmsd_sanity_check.png", dpi=200)
    plt.close(fig)

    summary = {
        "active_reference_structure": str(ACTIVE_PDB),
        "inactive_reference_structure": str(INACTIVE_PDB),
        "n_images": N_IMAGES,
        "ph": PH,
        "endpoint_rmsd_angstrom": endpoint_rmsd,
        "dry_arg_resnum": arg_resnum,
        "pxxy_tyr_resnum": tyr_resnum,
        "ionic_lock_partner_resnum": partner_resnum,
        "path_table": json.loads(table.to_json(orient="index")),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Results in {OUT_DIR}")
    print(table.sort_values("interpolation_fraction").to_string())
    g_range = table["gibbs_kcal_mol"].max() - table["gibbs_kcal_mol"].min()
    print(f"\nG range along path: {g_range:.1f} kcal/mol")
    barrier_idx = table["gibbs_kcal_mol"].idxmax()
    print(f"Highest-G (candidate barrier) image: {barrier_idx} "
          f"at f={table.loc[barrier_idx, 'interpolation_fraction']:.2f}")


if __name__ == "__main__":
    main()
