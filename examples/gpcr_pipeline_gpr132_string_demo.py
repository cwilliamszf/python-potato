"""
Real end-to-end interpolated-path pipeline run on GPR132 (G2A), the same
approach used for GPR68 in gpcr_pipeline_gpr68_string_demo.py /
gpcr_pipeline_gpr68_string_ph6_demo.py. GPR132 is another proton-sensing
class-A GPCR.

The receptor-agnostic pipeline logic (conserved-motif detection, linear
interpolation path building, real Gibbs scoring) is reused directly from
the GPR68 scripts via import -- nothing GPR68-specific was in it.

One thing IS new here and specific to this receptor's input data: the
active/inactive GPR132 models' N-terminus (residues 1-29) and C-terminus
(residues 356-380) disagree by 10-72 Angstrom between the two independent
models, while the ordered core (residues 30-355) disagrees by a physically
normal ~2.6 Angstrom Calpha RMSD once superposed on that core alone (see
examples/output/gpr132_string_demo/README.md for the full diagnostic).
This is the classic signature of an intrinsically disordered
terminus/tail placed essentially arbitrarily by each independent
AlphaFold-based model, not a real conformational difference -- so, unlike
the GPR68 run, this script trims both structures to the well-ordered core
(residues 30-355) before doing anything else. Interpolating the raw,
untrimmed termini would produce physically meaningless intermediate
structures (atoms displaced tens of Angstrom in a straight line, likely
straight through the folded core) and correspondingly meaningless
energies.

What's real vs. substituted, same disclosure as every other run this
session:
  * Tool 1 (protonation from pKa + pH): REAL -- PROPKA3 on the real,
    core-trimmed GPR132 active-state structure.
  * Tool 2 (AlphaFold/ColabFold ensemble): N/A for this script -- this is
    the interpolated-path alternative to tool 2, not tool 2 itself.
  * Tool 3 (Gibbs free energy): REAL -- gibbs/gpcr_gibbs_energy.py's
    actual AMBER ff14SB + GBn2 + RRHO pipeline via OpenMM/PDBFixer.
  * Tool 4 (landscape): available but, per the GPR68 string-demo's own
    finding, not the right tool for a handful of individually-relaxed
    path images (see that run's README) -- the direct G-vs-path plot is
    used here instead.

Run with (~15-20 min, 11 real OpenMM minimizations at pH 7.4):

    PYTHONPATH=.:gibbs python examples/gpcr_pipeline_gpr132_string_demo.py
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
from Bio.PDB import PDBParser
from Bio.PDB.Superimposer import Superimposer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gibbs"))

from gpcr_energy_landscapes.collective_variables import rmsd as cv_rmsd  # noqa: E402
from examples.gpcr_pipeline_common import trim_structure_to_residue_range, write_structure  # noqa: E402
from examples.gpcr_pipeline_gpr68_demo import find_conserved_motifs, find_ionic_lock_partner, run_tool1_protonation  # noqa: E402
from examples.gpcr_pipeline_gpr68_string_demo import build_interpolated_path, run_tool3_gibbs  # noqa: E402
from gpcr_ensemble.activation_state import tm3_tm6_distance  # noqa: E402

DATA_DIR = REPO_ROOT / "examples" / "data" / "gpr132_structures"
ACTIVE_PDB_RAW = DATA_DIR / "active" / "ClassA_gp132_human_Active_AF_2024-05-15_GPCRdb.pdb"
INACTIVE_PDB_RAW = DATA_DIR / "inactive" / "ClassA_gp132_human_Inactive_AFMS_2024-05-15_GPCRdb.pdb"
CORE_RESIDUE_RANGE = (30, 355)  # excludes disordered N-term (1-29) and C-tail (356-380); see README

OUT_DIR = Path(__file__).resolve().parent / "output" / "gpr132_string_demo"
N_IMAGES = 11
PH = 7.4


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = PDBParser(QUIET=True)
    active_raw = parser.get_structure("gpr132_active_raw", str(ACTIVE_PDB_RAW))
    inactive_raw = parser.get_structure("gpr132_inactive_raw", str(INACTIVE_PDB_RAW))

    a_res_raw = [r for r in active_raw[0]["A"] if r.id[0] == " "]
    i_res_raw = [r for r in inactive_raw[0]["A"] if r.id[0] == " "]
    assert len(a_res_raw) == len(i_res_raw), "active/inactive residue counts differ"
    for ra, ri in zip(a_res_raw, i_res_raw):
        assert ra.id[1] == ri.id[1] and {a.get_name() for a in ra} == {a.get_name() for a in ri}
    print(f"Verified exact atom correspondence: {len(a_res_raw)}/{len(a_res_raw)} residues match (untrimmed).")

    a_ca_raw = np.array([r["CA"].coord for r in a_res_raw])
    i_ca_raw = np.array([r["CA"].coord for r in i_res_raw])
    raw_rmsd = float(np.sqrt(((a_ca_raw - i_ca_raw) ** 2).sum(axis=1).mean()))
    print(f"Raw (untrimmed, no superposition) CA RMSD: {raw_rmsd:.2f} A "
          f"-- large due to disordered termini, see README before trusting this number")

    lo, hi = CORE_RESIDUE_RANGE
    print(f"\nTrimming both structures to the well-ordered core, residues {lo}-{hi}...")
    active_structure = trim_structure_to_residue_range(active_raw, "A", lo, hi)
    inactive_structure = trim_structure_to_residue_range(inactive_raw, "A", lo, hi)

    a_res = [r for r in active_structure[0]["A"] if r.id[0] == " "]
    i_res = [r for r in inactive_structure[0]["A"] if r.id[0] == " "]
    a_ca = np.array([r["CA"].coord for r in a_res])
    i_ca_before = np.array([r["CA"].coord for r in i_res])
    core_rmsd_raw_frame = float(np.sqrt(((a_ca - i_ca_before) ** 2).sum(axis=1).mean()))

    # Unlike GPR68 (where both endpoint models were already in the same
    # reference frame), GPR132's active/inactive models are NOT co-registered
    # -- see README: raw core RMSD is 18.99 A but drops to ~2.6 A after a
    # rigid-body fit. Linearly (Cartesian) interpolating two structures that
    # differ mainly by an unremoved rotation/translation, rather than a real
    # internal conformational change, would produce physically meaningless,
    # badly distorted intermediates (linear blending of a rotation is not a
    # rotation). So the inactive structure is rigidly superposed onto the
    # active structure's frame -- fit computed on the ordered core only, then
    # applied to every atom -- before interpolation.
    sup = Superimposer()
    sup.set_atoms([r["CA"] for r in a_res], [r["CA"] for r in i_res])
    sup.apply(list(inactive_structure.get_atoms()))
    print(f"Core ({len(a_res)} residues) CA RMSD before rigid-body superposition: {core_rmsd_raw_frame:.2f} A")
    print(f"Core CA RMSD after superposing inactive onto active's frame: {sup.rms:.2f} A "
          f"-- THIS is the physically meaningful conformational difference to interpolate")

    write_structure(active_structure, OUT_DIR / "active_core_trimmed.pdb")
    write_structure(inactive_structure, OUT_DIR / "inactive_core_superposed.pdb")
    ACTIVE_PDB = OUT_DIR / "active_core_trimmed.pdb"

    i_res = [r for r in inactive_structure[0]["A"] if r.id[0] == " "]
    i_ca = np.array([r["CA"].coord for r in i_res])

    arg_resnum, tyr_resnum = find_conserved_motifs(active_structure)
    partner_resnum = find_ionic_lock_partner(active_structure, arg_resnum, exclude_resnums=(tyr_resnum,))
    print(f"GPR132 auto-detected DRY-motif Arg (TM3, 3.50): resnum {arg_resnum}")
    print(f"GPR132 auto-detected TM7 P-x-x-Y motif Tyr (7.53): resnum {tyr_resnum}")
    print(f"GPR132 auto-detected ionic-lock partner (nearest Asp/Glu): resnum {partner_resnum}")

    pka = run_tool1_protonation(active_structure, ACTIVE_PDB)

    print(f"\nBuilding {N_IMAGES}-image linear interpolation path across the core region...")
    image_paths = build_interpolated_path(active_structure, inactive_structure, "A", N_IMAGES, OUT_DIR / "path")

    print(f"\nRunning real Gibbs free-energy calculations along the path at pH {PH} "
          f"({N_IMAGES} images, ~40-70s each -- core is smaller than GPR68's full receptor)...")
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
        rmsd_to_active[sid] = cv_rmsd(image_ca, a_ca)
        rmsd_to_inactive[sid] = cv_rmsd(image_ca, i_ca)

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

    order = table.sort_values("interpolation_fraction")
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(order["interpolation_fraction"], order["gibbs_kcal_mol"], "o-", lw=2, color="#3b6fa0")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("G (kcal/mol)")
    ax.set_title(f"GPR132: real Gibbs energy along active<->inactive core\ninterpolation path (pH {PH}, N={N_IMAGES} images)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gibbs_vs_path.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(order["interpolation_fraction"], order["rmsd_to_active"], "o-", label="RMSD to active", color="#3b6fa0")
    ax.plot(order["interpolation_fraction"], order["rmsd_to_inactive"], "o-", label="RMSD to inactive", color="#c0533e")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("Core CA RMSD (Å) after relaxation")
    ax.legend(frameon=False)
    ax.set_title("Did relaxation preserve the interpolated path, or collapse it?", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rmsd_sanity_check.png", dpi=200)
    plt.close(fig)

    summary = {
        "active_reference_structure_raw": str(ACTIVE_PDB_RAW),
        "inactive_reference_structure_raw": str(INACTIVE_PDB_RAW),
        "core_residue_range": list(CORE_RESIDUE_RANGE),
        "n_images": N_IMAGES,
        "ph": PH,
        "raw_untrimmed_rmsd_angstrom": raw_rmsd,
        "core_rmsd_original_frame_angstrom": core_rmsd_raw_frame,
        "core_rmsd_best_fit_angstrom": float(sup.rms),
        "dry_arg_resnum": arg_resnum,
        "pxxy_tyr_resnum": tyr_resnum,
        "ionic_lock_partner_resnum": partner_resnum,
        "pka_sample": dict(list(sorted(pka.items()))[:30]),
        "path_table": json.loads(table.to_json(orient="index")),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Results in {OUT_DIR}")
    print(table.sort_values("interpolation_fraction").to_string())
    barrier_idx = table["gibbs_kcal_mol"].idxmax()
    lower_endpoint = min(table["gibbs_kcal_mol"].iloc[0], table["gibbs_kcal_mol"].iloc[-1])
    barrier = table.loc[barrier_idx, "gibbs_kcal_mol"] - lower_endpoint
    print(f"Highest-G (candidate barrier) image: {barrier_idx} "
          f"at f={table.loc[barrier_idx, 'interpolation_fraction']:.2f}, barrier={barrier:.1f} kcal/mol")


if __name__ == "__main__":
    main()
