"""Regression/fidelity gate for xi calibration (Prompt 1, item 4).

Runs against real structures from the paper's own deposited dataset
(Anantakrishnan & Naganathan, Nat Commun 14:128 (2023);
github.com/AthiNaganathan/GPCR-Landscapes) -- NOT downloaded from PDB
directly (this sandbox's network access is blocked for every external
host tried, RCSB included), but sourced from that reference repo's own
bundled ``.mat``/``PBDs.zip`` files, which include the paper's own
reported PDBID/ene/Tm per receptor as ground truth
(``PDBID_gpcrNi``/``ene_gpcrNi``/``Tm_gpcrNi`` variables). See
``examples/data/gpcr_landscapes_reference/`` for the 5 real structures
this script uses; the mapping to real PDB IDs and the paper's own
reported ene/Tm (extracted directly from the reference repo's .mat
files, not estimated) is in ``REFERENCE_RECEPTORS`` below.

Two tiers, both against real receptor structures:

  Tier 1 (fast, all receptors): does this port's own Cp(T)/Tm machinery
  reproduce the paper's reported Tm when run AT the paper's own reported
  ene? This validates the ported physics (contact map, blocking,
  partition function, Cp derivative) independently of the Brent solver.

  Tier 2 (slow -- Brent search, ~15-20 min per receptor on this port's
  block counts): does this port's OWN calibrate_xi_tm_mode, run blind
  (no knowledge of the paper's reported ene), independently recover a
  comparable value? This validates the full solver end-to-end. Only run
  on a subset (default: rhodopsin + one more) given the cost.

Per Prompt 1: do not proceed to GPR68 receptor results until this gate
passes. Some numeric drift from the paper's own values is expected and
tolerated (see ``TM_TOLERANCE_K``/``ENE_TOLERANCE_J_MOL`` below): this
port's contact map, secondary-structure assignment, and blocking are
independently re-derived (not literally identical to the original
MATLAB code's DSSP-based assignment), verified in test_wsme_engine.py
et al. to reproduce the physics faithfully but not guaranteed
bit-identical.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsme_gpcr.calibration import (
    PAPER_TARGET_TM_K,
    PAPER_XI_MEAN_J_MOL,
    PAPER_XI_STD_J_MOL,
    calibrate_xi_tm_mode,
    compute_fc,
    find_cp_peaks_and_tm,
)
from wsme_gpcr.coupling import compute_coupling
from wsme_gpcr.dsc import compute_dsc
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams

HERE = Path(__file__).parent
REFDIR = HERE / "data" / "gpcr_landscapes_reference"

TM_TOLERANCE_K = 4.0
ENE_TOLERANCE_J_MOL = 10.0
FC_PAPER_MEAN_PCT = 13.0
FC_PAPER_STD_PCT = 4.5


@dataclass
class ReferenceReceptor:
    tag: str          # the paper's own generic file tag, e.g. "gpcr1i"
    pdb_id: str        # real PDB ID, from PDBID_<tag> in the reference .mat file
    paper_ene_kj_mol: float  # from ene_<tag>
    paper_tm_k: float        # from Tm_<tag>


# Extracted directly from GPCR-Landscapes' gpcr*.mat files
# (PDBID_<tag>/ene_<tag>/Tm_<tag>), not estimated or recalled from memory.
REFERENCE_RECEPTORS = [
    ReferenceReceptor("gpcr1i", "1U19", -0.0482, 333.0),   # rhodopsin (inactive) -- Prompt 1's named starting point
    ReferenceReceptor("gpcr2i", "2LNL", -0.0563, 333.0),   # largest-magnitude ene in this subset
    ReferenceReceptor("gpcr20i", "5LWE", -0.0452, 333.0),  # smallest-magnitude ene in this subset
    ReferenceReceptor("gpcr9i", "4DKL", -0.0499, 333.0),   # mu-opioid receptor (inactive)
    ReferenceReceptor("gpcr13a", "6OS9", -0.0550, 333.0),  # an active-state structure
]

TIER2_TAGS = {"gpcr1i", "gpcr9i"}  # Brent search only run on these (cost)


def tier1_check(receptor: ReferenceReceptor) -> dict:
    """At the paper's own reported ene, does this port's Cp(T)/Tm
    machinery reproduce their reported Tm?"""
    pdb_path = REFDIR / f"{receptor.tag}.pdb"
    result = run_pipeline(pdb_path, ph=7.0)
    params = WSMEParams(**{**result.params.__dict__, "ene": receptor.paper_ene_kj_mol})
    T_grid = np.arange(280.0, 360.0 + 1e-9, 0.5)
    dsc = compute_dsc(result.structure, result.block_model, result.ss_mask, params, T_grid=T_grid)
    tm_result = find_cp_peaks_and_tm(dsc.T, dsc.Cp_excess)
    delta = abs(tm_result.tm - receptor.paper_tm_k)
    return {
        "tag": receptor.tag, "pdb_id": receptor.pdb_id, "nblocks": result.block_model.nblocks,
        "paper_tm_k": receptor.paper_tm_k, "reproduced_tm_k": tm_result.tm,
        "delta_k": delta, "passed": delta <= TM_TOLERANCE_K, "is_bimodal": tm_result.is_bimodal,
    }


def tier2_check(receptor: ReferenceReceptor) -> dict:
    """Blind Brent-search calibration -- does this port's OWN solver
    recover a comparable ene, with no knowledge of the paper's value?"""
    pdb_path = REFDIR / f"{receptor.tag}.pdb"
    result = run_pipeline(pdb_path, ph=7.0)
    calib = calibrate_xi_tm_mode(result.structure, result.block_model, result.ss_mask,
                                  structure_path=pdb_path)
    delta_j_mol = abs(calib.xi_j_mol - receptor.paper_ene_kj_mol * 1000.0)

    coupling = compute_coupling(result.structure, result.block_model, result.ss_mask,
                                 params=WSMEParams(**{**WSMEParams().__dict__, "ene": calib.xi_kj_mol, "T": 310.0}))
    fc = 100.0 * compute_fc(coupling, result.block_model)

    return {
        "tag": receptor.tag, "pdb_id": receptor.pdb_id,
        "paper_ene_j_mol": receptor.paper_ene_kj_mol * 1000.0,
        "recovered_ene_j_mol": calib.xi_j_mol, "delta_j_mol": delta_j_mol,
        "passed_vs_paper": delta_j_mol <= ENE_TOLERANCE_J_MOL,
        "in_paper_bracket": -58.0 <= calib.xi_j_mol <= -40.0,
        "z_score": calib.z_score_vs_paper,
        "tm_achieved_k": calib.tm_achieved_k,
        "folded_minimum_ok": calib.folded_minimum_ok,
        "folded_minimum_frac": calib.folded_minimum_frac,
        "fc_pct": fc,
        "fc_within_paper_range": abs(fc - FC_PAPER_MEAN_PCT) <= 2 * FC_PAPER_STD_PCT,
    }


def main():
    print("=" * 78)
    print("TIER 1: Cp(T)/Tm reproduction at the paper's own reported ene")
    print("=" * 78)
    t0 = time.time()
    tier1_results = []
    for r in REFERENCE_RECEPTORS:
        t1 = time.time()
        res = tier1_check(r)
        tier1_results.append(res)
        status = "PASS" if res["passed"] else "FAIL"
        print(f"[{status}] {res['tag']} ({res['pdb_id']}, nblocks={res['nblocks']}): "
              f"paper Tm={res['paper_tm_k']:.0f} K, reproduced={res['reproduced_tm_k']:.1f} K "
              f"(delta={res['delta_k']:.1f} K, bimodal={res['is_bimodal']}) "
              f"(t={time.time()-t1:.1f}s)", flush=True)

    tier1_pass = all(r["passed"] for r in tier1_results)
    print(f"\nTier 1: {'PASS' if tier1_pass else 'FAIL'} "
          f"({sum(r['passed'] for r in tier1_results)}/{len(tier1_results)}) "
          f"(t={time.time()-t0:.1f}s)\n", flush=True)

    print("=" * 78)
    print("TIER 2: blind Brent-search xi recovery (calibrate_xi_tm_mode)")
    print("=" * 78)
    tier2_results = []
    for r in REFERENCE_RECEPTORS:
        if r.tag not in TIER2_TAGS:
            continue
        t2 = time.time()
        res = tier2_check(r)
        tier2_results.append(res)
        status = "PASS" if res["passed_vs_paper"] else "FAIL"
        print(f"[{status}] {res['tag']} ({res['pdb_id']}): paper ene={res['paper_ene_j_mol']:.2f} J/mol, "
              f"recovered={res['recovered_ene_j_mol']:.2f} J/mol (delta={res['delta_j_mol']:.2f} J/mol, "
              f"z={res['z_score']:+.2f}), Tm achieved={res['tm_achieved_k']:.1f} K, "
              f"in_paper_bracket={res['in_paper_bracket']}, "
              f"folded_min_ok={res['folded_minimum_ok']} ({res['folded_minimum_frac']:.1%}), "
              f"fc={res['fc_pct']:.1f}% (paper 13.0+/-4.5%, within_range={res['fc_within_paper_range']}) "
              f"(t={time.time()-t2:.1f}s)", flush=True)

    tier2_pass = all(r["passed_vs_paper"] and r["in_paper_bracket"] and r["folded_minimum_ok"]
                      for r in tier2_results) if tier2_results else False
    print(f"\nTier 2: {'PASS' if tier2_pass else 'FAIL'} "
          f"({sum(r['passed_vs_paper'] for r in tier2_results)}/{len(tier2_results)})\n", flush=True)

    overall = tier1_pass and tier2_pass
    print("=" * 78)
    print(f"FIDELITY GATE: {'PASS' if overall else 'FAIL'} -- "
          f"{'proceed to receptor results' if overall else 'DO NOT proceed to receptor results per Prompt 1'}")
    print("=" * 78)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
