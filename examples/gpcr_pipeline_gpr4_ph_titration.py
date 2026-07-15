"""
Task #16: fast, no-OpenMM real-structure pH-titration diagnostic across
all 7 real active/intermediate GPR4 cryo-EM structures the user
uploaded (9LGM pH8.0, 8ZCF pH7.5, 9JFX pH7.5, 9JFZ intermediate pH7.5,
9JFV pH6.8, 8ZCE pH6.0, 9BIP pH unspecified).

Unlike every other pH comparison in this repo (which is N=1 per
(structure, pH) because it compares Gibbs-scored path IMAGES built from
the same two endpoints at different scored pH), this is genuinely
different data: 7 INDEPENDENTLY SOLVED cryo-EM structures, several at
the *same* nominal pH (7.5: 8ZCF, 9JFX, 9JFZ) -- giving a real, if small,
look at structure-to-structure variability at fixed pH, not just a
single point. Report this distinction plainly; don't claim it as formal
error bars (n=2-3 is too small for that), but it IS more information
than a single structure would give.

Method: identify TM1-7 (via gpcr_pipeline_tm_topology.identify_tm_helices,
same DSSP+motif-anchored protocol validated on GPR68/GPR132) once from
the reference structure (8ZCF, arbitrary pick among the 7 -- all share
identical native numbering, verified in the extraction step), then use
those same TM ranges for all 7 (valid because of that shared numbering --
recomputing per-structure would only be necessary if numbering differed).
Each of the other 6 structures is Kabsch-superposed onto 8ZCF's frame
using the invariant core (TM1,2,3,4,5,7, excluding TM6), and TM6
cytoplasmic-tip displacement is measured relative to 8ZCF post-fit --
the same activation metric used throughout this project. Independent
cross-check: outlier-rejecting superposition with no TM6 prior.

No Gibbs/OpenMM step here -- this is real experimental geometry only,
answering "does TM6 position track experimental pH across independently
solved structures" without waiting on the expensive minimization runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from gpcr_pipeline_tm_topology import (
    identify_tm_helices,
    invariant_core_ca_atoms,
    iterative_outlier_rejecting_superposition,
    per_helix_rmsd,
    superpose_invariant_core,
    tm6_cytoplasmic_displacement,
)

DATA_ROOT = Path(__file__).parent / "data" / "gpr4_structures" / "clean_active"
OUT_DIR = Path(__file__).parent / "output" / "gpr4_ph_titration"
CHAIN_ID = "R"
REFERENCE_KEY = "8ZCF_pH7.5"

# (name, pdb filename, real experimental pH, or None if unspecified in the deposition)
STRUCTURES = [
    ("9LGM", "9LGM_pH8.0.pdb", 8.0),
    ("8ZCF", "8ZCF_pH7.5.pdb", 7.5),
    ("9JFX", "9JFX_pH7.5.pdb", 7.5),
    ("9JFZ_intermediate", "9JFZ_intermediate_pH7.5.pdb", 7.5),
    ("9JFV", "9JFV_pH6.8.pdb", 6.8),
    ("8ZCE", "8ZCE_pH6.0.pdb", 6.0),
    ("9BIP", "9BIP_pHunspecified.pdb", None),
]


def load(name):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(name, str(DATA_ROOT / name))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    structures = {name: load(fname) for name, fname, _ in STRUCTURES}
    ph_by_name = {name: ph for name, _, ph in STRUCTURES}

    reference = structures[REFERENCE_KEY.split("_")[0] if False else "8ZCF"]
    print(f"=== Identifying TM1-7 from reference structure ({REFERENCE_KEY}) ===")
    tm_ranges, anchors = identify_tm_helices(DATA_ROOT / "8ZCF_pH7.5.pdb", chain_id=CHAIN_ID)
    print(f"TM ranges: {tm_ranges}")
    print(f"DRY anchor: {anchors['dry']}, NPxxY anchor: {anchors['npxxy']}")

    core_atom_count = len(invariant_core_ca_atoms(reference, CHAIN_ID, tm_ranges, exclude=(6,)))
    print(f"Invariant core (TM1,2,3,4,5,7): {core_atom_count} CA atoms")

    results = []
    for name, fname, ph in STRUCTURES:
        moving = load(fname)  # fresh copy each time (superpose mutates in place)

        if name == "8ZCF":
            core_rmsd = 0.0
            tm6_disp, tip_resnums = 0.0, None
            per_helix = {tm: 0.0 for tm in tm_ranges}
            outlier_kept_frac = 1.0
            outlier_rms = 0.0
        else:
            sup = superpose_invariant_core(reference, moving, CHAIN_ID, tm_ranges, exclude=(6,))
            core_rmsd = sup.rms
            tm6_disp, tip_resnums = tm6_cytoplasmic_displacement(reference, moving, CHAIN_ID, tm_ranges)
            per_helix = per_helix_rmsd(reference, moving, CHAIN_ID, tm_ranges)

            # independent cross-check: fresh unmutated copies, no TM6 prior
            fresh_moving = load(fname)
            kept, outlier_rms = iterative_outlier_rejecting_superposition(
                reference, fresh_moving, CHAIN_ID, cutoff_ang=3.0
            )
            all_common = len(set(r.id[1] for r in reference[0][CHAIN_ID] if r.id[0] == " ")
                              & set(r.id[1] for r in fresh_moving[0][CHAIN_ID] if r.id[0] == " "))
            outlier_kept_frac = len(kept) / all_common if all_common else float("nan")

        row = {
            "name": name,
            "ph": ph,
            "invariant_core_rmsd_angstrom": round(core_rmsd, 4),
            "tm6_cytoplasmic_displacement_angstrom": round(tm6_disp, 4),
            "per_helix_rmsd": {k: round(v, 4) for k, v in per_helix.items()},
            "outlier_rejecting_kept_fraction": round(outlier_kept_frac, 4),
            "outlier_rejecting_rms_angstrom": round(outlier_rms, 4),
        }
        results.append(row)
        print(f"{name:20s} pH={ph!s:>5}  core_rmsd={core_rmsd:7.4f}  "
              f"TM6_disp={tm6_disp:7.4f}  outlier_kept_frac={outlier_kept_frac:.3f}")

    summary = {
        "reference": REFERENCE_KEY,
        "tm_ranges": {str(k): v for k, v in tm_ranges.items()},
        "anchors": {"dry": anchors["dry"], "npxxy": anchors["npxxy"]},
        "invariant_core_atom_count": core_atom_count,
        "results": results,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot: TM6 displacement (relative to 8ZCF) vs experimental pH
    plotted = [r for r in results if r["ph"] is not None]
    plotted.sort(key=lambda r: r["ph"])
    phs = [r["ph"] for r in plotted]
    disps = [r["tm6_cytoplasmic_displacement_angstrom"] for r in plotted]
    names = [r["name"] for r in plotted]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(phs, disps, s=60, zorder=3)
    for x, y, label in zip(phs, disps, names):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Experimental pH (as deposited)")
    ax.set_ylabel(f"TM6 cytoplasmic-tip displacement vs. {REFERENCE_KEY} (Å)")
    ax.set_title("GPR4: real-structure TM6 position vs. experimental pH\n"
                  "(invariant-core superposition, no OpenMM/Gibbs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tm6_displacement_vs_ph.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote {OUT_DIR / 'summary.json'} and {OUT_DIR / 'tm6_displacement_vs_ph.png'}")


if __name__ == "__main__":
    main()
