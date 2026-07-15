"""
Task #17/#18: symmetric pipeline diagnostics + real Gibbs pipeline for
GPR4, using the same one-protocol-no-exceptions method validated on
GPR68 and GPR132 (see gpcr_pipeline_symmetric_string_demo.py's module
docstring for the full method rationale), applied to REAL experimental
cryo-EM structures for the first time in this repo rather than
AlphaFold-based homology models.

Two differences from the GPR68/GPR132 script, both forced by working
with independently-solved real structures rather than models from one
shared pipeline:

1. Chain ID is "R" (all 11 uploaded structures use this), not "A".
2. GPR68/GPR132's active/inactive endpoints share IDENTICAL residue
   coverage (same homology-modeling pipeline/template) -- this script's
   predecessor could require an exact resnum-list match and treat any
   violation as an error. Real, independently-solved cryo-EM structures
   don't: 9JFU (inactive, BRIL-fusion-excised and alignment-renumbered
   in gpcr_pipeline_cryoem_extract.py) has two small real gaps (a
   1-residue gap at 63, a 6-residue disordered-loop gap at 211-216)
   that the active structures don't share. Every step here is rewritten
   to operate on the INTERSECTION of resolved resnums rather than
   assuming identical coverage -- see `superpose_invariant_core_common`
   (gpcr_pipeline_tm_topology.py) and `build_interpolated_path_common`
   below. Within the TM1-TM7 core span, coverage overlap is 261/268
   residues (97.4%) for both active endpoints -- verified before
   running anything expensive.

Two pH-matched real-structure paths, chosen specifically because they
use two INDEPENDENTLY SOLVED active-state structures at two different
real experimental pH values, rather than reusing one active structure
and only varying scored pH (which is what the GPR68/GPR132 runs did,
for lack of real pH-resolved structures):

  Path A: 9JFU (inactive, antagonist-bound) -> 8ZCF (active, pH 7.5),
          scored at pH 7.4 (matches the deposition's near-physiological pH).
  Path B: 9JFU (inactive, antagonist-bound) -> 8ZCE (active, pH 6.0),
          scored at pH 6.0 (matches the deposition's pH exactly).

Same sanity gates as before: TM6 cytoplasmic displacement >= 4 A to
count as a genuine activation pair, N=1 per (image, path) with no error
bars, Gate A (uncalibrated absolute PROPKA/Gibbs energetics) still open.

Run with (~15-20 min for diagnostics, plus ~35-45 min for the 22 real
Gibbs calculations):

    PYTHONPATH=.:gibbs python examples/gpcr_pipeline_gpr4_symmetric_demo.py
"""

from __future__ import annotations

import json
import subprocess
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

import gpcr_gibbs_energy as gibbs_tool  # noqa: E402
from examples.gpcr_pipeline_common import trim_structure_to_residue_range, write_structure  # noqa: E402
from examples.gpcr_pipeline_tm_topology import (  # noqa: E402
    identify_tm_helices, iterative_outlier_rejecting_superposition,
    per_helix_rmsd, superpose_invariant_core_common, tm6_cytoplasmic_displacement,
)
from gpcr_energy_landscapes.collective_variables import rmsd as cv_rmsd  # noqa: E402

DATA_ROOT = REPO_ROOT / "examples" / "data" / "gpr4_structures"
CHAIN_ID = "R"
INACTIVE_PDB = DATA_ROOT / "clean_inactive" / "9JFU_clean.pdb"
PATHS = {
    "path_a_ph7.4": {"active_pdb": DATA_ROOT / "clean_active" / "8ZCF_pH7.5.pdb", "ph": 7.4,
                      "active_label": "8ZCF (pH 7.5 deposition)"},
    "path_b_ph6.0": {"active_pdb": DATA_ROOT / "clean_active" / "8ZCE_pH6.0.pdb", "ph": 6.0,
                      "active_label": "8ZCE (pH 6.0 deposition)"},
}

N_IMAGES = 11
TM6_GATE_ANGSTROM = 4.0
OUTLIER_CUTOFF_ANGSTROM = 3.0
MINIMIZE_ITERATIONS = 150

OUT_ROOT = Path(__file__).resolve().parent / "output" / "gpr4_symmetric_demo"


def run_tool3_gibbs(pdb_path, ph, chain_id=CHAIN_ID):
    """Like gpcr_pipeline_gpr68_string_demo.run_tool3_gibbs, but with a
    configurable chain -- that script hardcodes "--chains A", which is
    wrong for these real cryo-EM structures (chain "R" throughout) and
    silently produces an empty/chainless structure that crashes deep
    inside PDBFixer/OpenMM's PDB parser rather than failing loudly at
    the chain-selection step."""
    argv = [
        str(pdb_path),
        "--chains", chain_id,
        "--ph", str(ph),
        "--entropy-method", "ca-anm",
        "--minimize-iterations", str(MINIMIZE_ITERATIONS),
        "--no-hbond-analysis",
    ]
    args = gibbs_tool.parse_args(argv)
    result = gibbs_tool.run(args)
    return result["G_kcal"]


def sha256_of(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_interpolated_path_common(active_structure, inactive_structure, chain_id, n_images, out_dir):
    """Resnum-matched (not position-matched) linear Cartesian
    interpolation for endpoints with different residue coverage: builds
    each image only over resnums resolved in BOTH structures, and within
    each shared residue, only over atom names present in BOTH (a
    real-structure residue can be missing e.g. a distal, weakly-resolved
    side-chain atom in one deposition and not the other -- silently
    skipping that one atom for that one residue is preferable to
    crashing the whole path, and the downstream minimizer rebuilds
    missing atoms as part of protonation/relaxation anyway)."""
    a_chain = active_structure[0][chain_id]
    i_chain = inactive_structure[0][chain_id]
    a_by_resnum = {r.id[1]: r for r in a_chain if r.id[0] == " "}
    i_by_resnum = {r.id[1]: r for r in i_chain if r.id[0] == " "}
    common_resnums = sorted(set(a_by_resnum) & set(i_by_resnum))

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    image_paths = {}
    fractions = np.linspace(0.0, 1.0, n_images)

    for k, f in enumerate(fractions):
        image_structure = active_structure.copy()
        image_chain = image_structure[0][chain_id]
        drop_residues = [r for r in image_chain if r.id[0] == " " and r.id[1] not in common_resnums]
        for r in drop_residues:
            image_chain.detach_child(r.id)

        for resnum in common_resnums:
            res_img = image_chain[(" ", resnum, " ")]
            res_a = a_by_resnum[resnum]
            res_i = i_by_resnum[resnum]
            common_atom_names = set(a.get_name() for a in res_a) & set(a.get_name() for a in res_i)
            drop_atoms = [a for a in res_img if a.get_name() not in common_atom_names]
            for a in drop_atoms:
                res_img.detach_child(a.get_id())
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

    return image_paths, common_resnums


def run_diagnostics(name, active_pdb, inactive_pdb, tm_ranges, anchors, out_dir):
    parser = PDBParser(QUIET=True)
    active_raw = parser.get_structure(f"{name}_active", str(active_pdb))
    inactive_raw = parser.get_structure(f"{name}_inactive", str(inactive_pdb))

    a_res = {r.id[1]: r for r in active_raw[0][CHAIN_ID] if r.id[0] == " "}
    i_res = {r.id[1]: r for r in inactive_raw[0][CHAIN_ID] if r.id[0] == " "}
    common_all = sorted(set(a_res) & set(i_res))
    a_ca_raw = np.array([a_res[rn]["CA"].coord for rn in common_all])
    i_ca_raw = np.array([i_res[rn]["CA"].coord for rn in common_all])
    raw_rmsd = float(np.sqrt(((a_ca_raw - i_ca_raw) ** 2).sum(axis=1).mean()))

    active = parser.get_structure(f"{name}_active2", str(active_pdb))
    inactive = parser.get_structure(f"{name}_inactive2", str(inactive_pdb))
    sup, common_core_n, active_core_n = superpose_invariant_core_common(
        active, inactive, CHAIN_ID, tm_ranges, exclude=(6,)
    )

    helix_rmsd = per_helix_rmsd(active, inactive, CHAIN_ID, tm_ranges)
    tm6_disp, tm6_tip_resnums = tm6_cytoplasmic_displacement(active, inactive, CHAIN_ID, tm_ranges)

    active_oc = parser.get_structure(f"{name}_active_oc", str(active_pdb))
    inactive_oc = parser.get_structure(f"{name}_inactive_oc", str(inactive_pdb))
    kept, outlier_rms = iterative_outlier_rejecting_superposition(
        active_oc, inactive_oc, CHAIN_ID, cutoff_ang=OUTLIER_CUTOFF_ANGSTROM
    )
    invariant_core_resnums = set()
    for tm, (lo, hi) in tm_ranges.items():
        if tm != 6:
            invariant_core_resnums.update(range(lo, hi + 1))
    overlap = len(set(kept) & invariant_core_resnums) / len(invariant_core_resnums)

    tm6_gate_passed = tm6_disp >= TM6_GATE_ANGSTROM

    diagnostics = {
        "pair": name,
        "active_pdb": str(active_pdb),
        "active_pdb_sha256_16": sha256_of(active_pdb),
        "inactive_pdb": str(inactive_pdb),
        "inactive_pdb_sha256_16": sha256_of(inactive_pdb),
        "tm_ranges": {f"TM{k}": list(v) for k, v in tm_ranges.items()},
        "dry_anchor": anchors["dry"],
        "npxxy_anchor": anchors["npxxy"],
        "raw_common_resnum_count": len(common_all),
        "raw_whole_chain_rmsd_angstrom": raw_rmsd,
        "invariant_core_common_resnum_count": common_core_n,
        "invariant_core_active_total_count": active_core_n,
        "invariant_core_coverage_fraction": common_core_n / active_core_n,
        "invariant_core_superposition_rms_angstrom": float(sup.rms),
        "per_helix_rmsd_post_superposition": {f"TM{k}": v for k, v in helix_rmsd.items()},
        "tm6_cytoplasmic_displacement_angstrom": tm6_disp,
        "tm6_tip_resnums": tm6_tip_resnums,
        "tm6_gate_4A_passed": tm6_gate_passed,
        "outlier_rejection_cutoff_angstrom": OUTLIER_CUTOFF_ANGSTROM,
        "outlier_rejection_kept_n": len(kept),
        "outlier_rejection_total_n": len(common_all),
        "outlier_rejection_rms_angstrom": float(outlier_rms),
        "outlier_vs_bw_core_overlap_fraction": overlap,
    }
    print(f"\n{'='*70}\n{name} DIAGNOSTICS\n{'='*70}")
    print(f"  Common resolved residues (active vs inactive): {len(common_all)}")
    print(f"  Raw whole-chain CA RMSD (no superposition, common resnums only): {raw_rmsd:.2f} A")
    print(f"  Invariant-core coverage: {common_core_n}/{active_core_n} "
          f"({common_core_n/active_core_n:.1%}) residues used for the fit")
    print(f"  Invariant-core (TM1,2,3,4,5,7) superposition RMS: {sup.rms:.3f} A")
    for tm in range(1, 8):
        print(f"    TM{tm} RMSD post-superposition: {helix_rmsd[tm]:.2f} A")
    print(f"  TM6 cytoplasmic-tip displacement: {tm6_disp:.2f} A "
          f"({'PASSES' if tm6_gate_passed else 'FAILS'} the {TM6_GATE_ANGSTROM} A genuine-activation-pair gate)")
    print(f"  Outlier-rejection cross-check (cutoff={OUTLIER_CUTOFF_ANGSTROM} A): "
          f"kept {len(kept)}/{len(common_all)}, rms={outlier_rms:.3f} A, "
          f"{overlap:.1%} overlap with BW-defined invariant core")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    return diagnostics, active, inactive


def run_one_path(name, active_pdb, ph, active_label, tm_ranges, anchors):
    out_dir = OUT_ROOT / name
    diagnostics, active, inactive = run_diagnostics(name, active_pdb, INACTIVE_PDB, tm_ranges, anchors, out_dir)

    core_lo = tm_ranges[1][0]
    core_hi = tm_ranges[7][1]
    print(f"\n  Interpolation core span (TM1 start .. TM7 end): {core_lo}-{core_hi}")

    active_core = trim_structure_to_residue_range(active, CHAIN_ID, core_lo, core_hi)
    inactive_core = trim_structure_to_residue_range(inactive, CHAIN_ID, core_lo, core_hi)
    write_structure(active_core, out_dir / "active_core.pdb")
    write_structure(inactive_core, out_dir / "inactive_core.pdb")

    print(f"\n  Building {N_IMAGES}-image linear interpolation path ({active_label} <-> 9JFU inactive)...")
    image_paths, common_resnums = build_interpolated_path_common(active_core, inactive_core, CHAIN_ID, N_IMAGES, out_dir / "path")
    print(f"  Path built over {len(common_resnums)} resnum-matched, atom-matched residues")

    a_ca = np.array([active_core[0][CHAIN_ID][(" ", rn, " ")]["CA"].coord for rn in common_resnums])
    i_ca = np.array([inactive_core[0][CHAIN_ID][(" ", rn, " ")]["CA"].coord for rn in common_resnums])
    core_endpoint_rmsd = float(cv_rmsd(a_ca, i_ca))
    print(f"  Core endpoint CA RMSD (common resnums, after invariant-core superposition): {core_endpoint_rmsd:.2f} A")

    parser = PDBParser(QUIET=True)
    print(f"\n  Running real Gibbs free-energy calculations for {name} at pH {ph} "
          f"({N_IMAGES} images, ~40-90s each)...")
    energies = {}
    rmsd_records = []
    t0 = time.time()
    for idx, (sid, (path, f)) in enumerate(image_paths.items()):
        g = run_tool3_gibbs(path, ph)
        energies[sid] = g

        conf = parser.get_structure(sid, str(path))
        chain = conf[0][CHAIN_ID]
        image_by_resnum = {r.id[1]: r for r in chain if r.id[0] == " "}
        image_ca = np.array([image_by_resnum[rn]["CA"].coord for rn in common_resnums if rn in image_by_resnum])
        r_active = float(cv_rmsd(image_ca, a_ca))
        r_inactive = float(cv_rmsd(image_ca, i_ca))
        rmsd_records.append({"structure_id": sid, "f": f, "rmsd_to_active": r_active, "rmsd_to_inactive": r_inactive})

        print(f"    [{idx+1}/{N_IMAGES}] {sid} (f={f:.2f}): G = {g:.2f} kcal/mol, "
              f"RMSD-to-active={r_active:.2f} A, RMSD-to-inactive={r_inactive:.2f} A "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    fractions = {sid: f for sid, (_, f) in image_paths.items()}
    table = pd.DataFrame({"structure_id": list(image_paths.keys()),
                           "interpolation_fraction": [fractions[s] for s in image_paths]}).set_index("structure_id")
    table[f"gibbs_kcal_mol_ph{ph}"] = pd.Series(energies)
    rmsd_df = pd.DataFrame(rmsd_records).set_index("structure_id")
    table = table.join(rmsd_df[["rmsd_to_active", "rmsd_to_inactive"]])
    table.to_csv(out_dir / "path_table.csv")

    order = table.sort_values("interpolation_fraction")
    col = f"gibbs_kcal_mol_ph{ph}"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(order["interpolation_fraction"], order[col], "o-", lw=2, color="#3b6fa0")
    ax.set_xlabel(f"Interpolation fraction (0={active_label}, 1=9JFU inactive)")
    ax.set_ylabel("G (kcal/mol)")
    gate_note = "PASSES" if diagnostics["tm6_gate_4A_passed"] else "FAILS (not a genuine activation pair)"
    ax.set_title(f"GPR4 {name}: G along real-structure path (N={N_IMAGES}, pH {ph})\n"
                 f"TM6 displacement={diagnostics['tm6_cytoplasmic_displacement_angstrom']:.2f} A -- {gate_note}", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "gibbs_vs_path.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(order["interpolation_fraction"], order["rmsd_to_active"], "o-", label="RMSD to active", color="#3b6fa0")
    ax.plot(order["interpolation_fraction"], order["rmsd_to_inactive"], "o-", label="RMSD to inactive", color="#c0533e")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("Core CA RMSD (Å) after relaxation")
    ax.legend(frameon=False)
    ax.set_title(f"GPR4 {name}: did relaxation preserve the interpolated path?", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "rmsd_sanity_check.png", dpi=200)
    plt.close(fig)

    peak_sid = order[col].idxmax()
    endpoint_active = float(order[col].iloc[0])
    endpoint_inactive = float(order[col].iloc[-1])
    lower_endpoint = min(endpoint_active, endpoint_inactive)
    barrier = float(order.loc[peak_sid, col]) - lower_endpoint

    summary = {
        "pair": name, "active_label": active_label, "ph": ph,
        "diagnostics": diagnostics, "core_span": [core_lo, core_hi],
        "core_endpoint_rmsd_angstrom": core_endpoint_rmsd,
        "n_common_path_residues": len(common_resnums),
        "n_images": N_IMAGES, "minimize_iterations": MINIMIZE_ITERATIONS,
        "endpoint_active_G": endpoint_active, "endpoint_inactive_G": endpoint_inactive,
        "endpoint_shift_inactive_minus_active": endpoint_inactive - endpoint_active,
        "peak_f": float(order.loc[peak_sid, "interpolation_fraction"]),
        "peak_G": float(order.loc[peak_sid, col]),
        "barrier_vs_lower_endpoint_kcal_mol": barrier,
        "config_hash": sha256_of(__file__), "git_commit": git_commit_hash(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{name} DONE. Results in {out_dir}")
    print(f"  Core endpoint RMSD: {core_endpoint_rmsd:.2f} A")
    print(f"  pH {ph}: barrier={barrier:.1f} kcal/mol at f={summary['peak_f']:.2f}, "
          f"endpoint shift (inactive-active)={summary['endpoint_shift_inactive_minus_active']:+.1f} kcal/mol")

    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=== Identifying TM1-7 (once, from 8ZCF -- shared numbering across all active structures) ===")
    reference_active_pdb = PATHS["path_a_ph7.4"]["active_pdb"]
    tm_ranges, anchors = identify_tm_helices(reference_active_pdb, chain_id=CHAIN_ID)
    print(f"TM ranges: {tm_ranges}")
    print(f"DRY: {anchors['dry']}, NPxxY: {anchors['npxxy']}")

    all_summaries = {}
    for name, cfg in PATHS.items():
        all_summaries[name] = run_one_path(name, cfg["active_pdb"], cfg["ph"], cfg["active_label"], tm_ranges, anchors)

    with open(OUT_ROOT / "all_paths_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    print(f"\n{'='*70}\nALL GPR4 PATHS DONE\n{'='*70}")
    for name, s in all_summaries.items():
        gate = "PASS" if s["diagnostics"]["tm6_gate_4A_passed"] else "FAIL"
        print(f"  {name} (pH {s['ph']}): TM6 disp={s['diagnostics']['tm6_cytoplasmic_displacement_angstrom']:.2f} A [{gate}], "
              f"barrier={s['barrier_vs_lower_endpoint_kcal_mol']:.1f} kcal/mol")


if __name__ == "__main__":
    main()
