"""
Corrected, symmetric interpolated-path pipeline -- replaces
gpcr_pipeline_gpr68_string_demo.py/_ph6_demo.py and
gpcr_pipeline_gpr132_string_demo.py/_ph6_demo.py, whose GPR68 and GPR132
runs used DIFFERENT geometric preprocessing (GPR68: no superposition at
all; GPR132: Kabsch fit on a manually-chosen residue range). That
asymmetry was flagged in review as confounding the very comparison the
two runs were meant to support, and the fix required treating "no
superposition" as itself a preprocessing choice, not a neutral default --
a raw two-file RMSD carries whatever incidental rotation/translation
offset the two files happen to have, which is a file-format artifact,
not biology.

ONE protocol, applied identically to every receptor with no exceptions:

  1. Identify TM1-TM7 by DSSP-based secondary structure (real H-bond
     geometry, not phi/psi-only heuristics -- see gpcr_pipeline_tm_topology's
     module docstring for why the phi/psi-only version was unreliable),
     anchored by the DRY-like (TM3) and NPxxY-like (TM7) motifs, with
     TM1/2/4/5/6 assigned by their invariant topological sequence order.
     Self-consistency (TM7 index == TM3 index + 4) is checked, not assumed.
  2. Superpose the inactive structure onto the active structure using ONLY
     TM1, TM2, TM3, TM4, TM5, TM7 backbone CA (TM6 -- the principal mover
     -- and all loops/termini excluded from the FIT, not from the
     structure). This removes any incidental frame offset while leaving
     TM6's real displacement, if any, fully intact as the signal to
     measure.
  3. Cross-check with an independent, prior-free iterative
     outlier-rejecting superposition (fit all CA, drop atoms beyond a
     fixed 3.0 A cutoff, refit, converge) and report the overlap between
     the two methods' surviving atom sets.
  4. Measure TM6 cytoplasmic-tip displacement post-superposition -- the
     hallmark class-A activation metric -- and gate on it: a pair with
     TM6 displacement < 4 A is not a genuine activation transition, is
     flagged loudly, and its pH-response result must not be interpreted
     as evidence about proton-sensing conformational switching.
  5. Trim both structures to the ordered receptor core (TM1's first
     residue through TM7's last residue -- derived from the same
     detection in step 1, not a separately hand-picked range) before
     interpolation, excluding only genuinely disordered termini outside
     the 7TM+H8 span.
  6. 11-image linear Cartesian interpolation, f = 0.0 (active) to 1.0
     (inactive), same as every prior run this session.
  7. Same minimizer (150 L-BFGS iterations) and scorer (AMBER ff14SB +
     GBn2 + RRHO Gibbs via OpenMM/PDBFixer, PROPKA-assigned protonation)
     as every prior run, at pH 7.4 and pH 6.0.
  8. RMSD-to-active / RMSD-to-inactive sanity check per path, as before.

Gate A (uncalibrated absolute pKa/energetics -- PROPKA has not been
validated against an experimental buried-carboxylate benchmark by this
pipeline) is still open. Absolute G magnitudes here are uncalibrated;
the differential structure along one path, computed identically for
both receptors, is the more defensible comparison. N=1 per (image, pH):
there are no error bars, and no difference reported here should be
described as statistically significant.

Run with (~50-70 min total: 2 receptors x 11 images x 2 pH values, each
a real OpenMM minimization):

    PYTHONPATH=.:gibbs python examples/gpcr_pipeline_symmetric_string_demo.py
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
from Bio.PDB import PDBParser

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gibbs"))

from examples.gpcr_pipeline_common import write_structure  # noqa: E402
from examples.gpcr_pipeline_gpr68_string_demo import build_interpolated_path, run_tool3_gibbs  # noqa: E402
from examples.gpcr_pipeline_tm_topology import (  # noqa: E402
    identify_tm_helices, invariant_core_ca_atoms, iterative_outlier_rejecting_superposition,
    per_helix_rmsd, superpose_invariant_core, tm6_cytoplasmic_displacement,
)
from gpcr_energy_landscapes.collective_variables import rmsd as cv_rmsd  # noqa: E402

DATA_ROOT = REPO_ROOT / "examples" / "data"
RECEPTORS = {
    "GPR68": {
        "active": DATA_ROOT / "gpr68_structures" / "active" / "ClassA_ogr1_human_Active_AFMS_2024-05-15_GPCRdb.pdb",
        "inactive": DATA_ROOT / "gpr68_structures" / "inactive" / "ClassA_ogr1_human_Inactive_AFMS_2024-05-15_GPCRdb.pdb",
    },
    "GPR132": {
        "active": DATA_ROOT / "gpr132_structures" / "active" / "ClassA_gp132_human_Active_AF_2024-05-15_GPCRdb.pdb",
        "inactive": DATA_ROOT / "gpr132_structures" / "inactive" / "ClassA_gp132_human_Inactive_AFMS_2024-05-15_GPCRdb.pdb",
    },
}

N_IMAGES = 11
PH_VALUES = [7.4, 6.0]
TM6_GATE_ANGSTROM = 4.0
OUTLIER_CUTOFF_ANGSTROM = 3.0
MINIMIZE_ITERATIONS = 150

OUT_ROOT = Path(__file__).resolve().parent / "output" / "symmetric_string_demo"


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


def run_diagnostics(name, active_pdb, inactive_pdb, out_dir):
    parser = PDBParser(QUIET=True)
    active_raw = parser.get_structure(f"{name}_active", str(active_pdb))
    inactive_raw = parser.get_structure(f"{name}_inactive", str(inactive_pdb))

    tm_ranges, anchors = identify_tm_helices(active_pdb)

    a_res = [r for r in active_raw[0]["A"] if r.id[0] == " "]
    i_res = [r for r in inactive_raw[0]["A"] if r.id[0] == " "]
    assert [r.id[1] for r in a_res] == [r.id[1] for r in i_res], f"{name}: residue numbering mismatch"
    a_ca_raw = np.array([r["CA"].coord for r in a_res])
    i_ca_raw = np.array([r["CA"].coord for r in i_res])
    raw_rmsd = float(np.sqrt(((a_ca_raw - i_ca_raw) ** 2).sum(axis=1).mean()))

    # Invariant-core (TM1,2,3,4,5,7) superposition -- inactive mutated in place
    active = parser.get_structure(f"{name}_active2", str(active_pdb))
    inactive = parser.get_structure(f"{name}_inactive2", str(inactive_pdb))
    sup = superpose_invariant_core(active, inactive, "A", tm_ranges, exclude=(6,))

    helix_rmsd = per_helix_rmsd(active, inactive, "A", tm_ranges)
    tm6_disp, tm6_tip_resnums = tm6_cytoplasmic_displacement(active, inactive, "A", tm_ranges)

    # Independent cross-check on fresh, unmutated copies
    active_oc = parser.get_structure(f"{name}_active_oc", str(active_pdb))
    inactive_oc = parser.get_structure(f"{name}_inactive_oc", str(inactive_pdb))
    kept, outlier_rms = iterative_outlier_rejecting_superposition(
        active_oc, inactive_oc, "A", cutoff_ang=OUTLIER_CUTOFF_ANGSTROM
    )
    invariant_core_resnums = set()
    for tm, (lo, hi) in tm_ranges.items():
        if tm != 6:
            invariant_core_resnums.update(range(lo, hi + 1))
    overlap = len(set(kept) & invariant_core_resnums) / len(invariant_core_resnums)

    tm6_gate_passed = tm6_disp >= TM6_GATE_ANGSTROM

    diagnostics = {
        "receptor": name,
        "active_pdb": str(active_pdb),
        "active_pdb_sha256_16": sha256_of(active_pdb),
        "inactive_pdb": str(inactive_pdb),
        "inactive_pdb_sha256_16": sha256_of(inactive_pdb),
        "tm_ranges": {f"TM{k}": list(v) for k, v in tm_ranges.items()},
        "dry_anchor": anchors["dry"],
        "npxxy_anchor": anchors["npxxy"],
        "raw_whole_chain_rmsd_angstrom": raw_rmsd,
        "invariant_core_superposition_rms_angstrom": float(sup.rms),
        "per_helix_rmsd_post_superposition": {f"TM{k}": v for k, v in helix_rmsd.items()},
        "tm6_cytoplasmic_displacement_angstrom": tm6_disp,
        "tm6_tip_resnums": tm6_tip_resnums,
        "tm6_gate_4A_passed": tm6_gate_passed,
        "outlier_rejection_cutoff_angstrom": OUTLIER_CUTOFF_ANGSTROM,
        "outlier_rejection_kept_n": len(kept),
        "outlier_rejection_total_n": len(a_res),
        "outlier_rejection_rms_angstrom": float(outlier_rms),
        "outlier_vs_bw_core_overlap_fraction": overlap,
    }
    print(f"\n{'='*70}\n{name} DIAGNOSTICS\n{'='*70}")
    print(f"  TM ranges: {diagnostics['tm_ranges']}")
    print(f"  Raw whole-chain CA RMSD (no superposition): {raw_rmsd:.2f} A")
    print(f"  Invariant-core (TM1,2,3,4,5,7) superposition RMS: {sup.rms:.3f} A")
    for tm in range(1, 8):
        print(f"    TM{tm} RMSD post-superposition: {helix_rmsd[tm]:.2f} A")
    print(f"  TM6 cytoplasmic-tip displacement: {tm6_disp:.2f} A "
          f"({'PASSES' if tm6_gate_passed else 'FAILS'} the {TM6_GATE_ANGSTROM} A genuine-activation-pair gate)")
    print(f"  Outlier-rejection cross-check (cutoff={OUTLIER_CUTOFF_ANGSTROM} A): "
          f"kept {len(kept)}/{len(a_res)}, rms={outlier_rms:.3f} A, "
          f"{overlap:.1%} overlap with BW-defined invariant core")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    return tm_ranges, anchors, diagnostics, active, inactive


def run_one_receptor(name, active_pdb, inactive_pdb):
    out_dir = OUT_ROOT / name.lower()
    tm_ranges, anchors, diagnostics, active, inactive = run_diagnostics(name, active_pdb, inactive_pdb, out_dir)

    # Trim to the ordered receptor core: TM1's first residue through TM7's last.
    core_lo = tm_ranges[1][0]
    core_hi = tm_ranges[7][1]
    print(f"\n  Interpolation core span (TM1 start .. TM7 end): {core_lo}-{core_hi}")

    from examples.gpcr_pipeline_common import trim_structure_to_residue_range

    active_core = trim_structure_to_residue_range(active, "A", core_lo, core_hi)
    inactive_core = trim_structure_to_residue_range(inactive, "A", core_lo, core_hi)
    write_structure(active_core, out_dir / "active_core.pdb")
    write_structure(inactive_core, out_dir / "inactive_core.pdb")

    a_res = [r for r in active_core[0]["A"] if r.id[0] == " "]
    i_res = [r for r in inactive_core[0]["A"] if r.id[0] == " "]
    a_ca = np.array([r["CA"].coord for r in a_res])
    i_ca = np.array([r["CA"].coord for r in i_res])
    core_endpoint_rmsd = float(cv_rmsd(a_ca, i_ca))
    print(f"  Core (TM1-TM7 span) endpoint CA RMSD after invariant-core superposition: {core_endpoint_rmsd:.2f} A")

    print(f"\n  Building {N_IMAGES}-image linear interpolation path...")
    image_paths = build_interpolated_path(active_core, inactive_core, "A", N_IMAGES, out_dir / "path")

    parser = PDBParser(QUIET=True)
    all_results = {}
    rmsd_records = []
    for ph in PH_VALUES:
        print(f"\n  Running real Gibbs free-energy calculations for {name} at pH {ph} "
              f"({N_IMAGES} images, ~40-90s each)...")
        energies = {}
        t0 = time.time()
        for idx, (sid, (path, f)) in enumerate(image_paths.items()):
            g = run_tool3_gibbs(path, ph)
            energies[sid] = g

            conf = parser.get_structure(sid, str(path))
            chain = conf[0]["A"]
            image_ca = np.array([r["CA"].coord for r in chain if r.id[0] == " "])
            r_active = float(cv_rmsd(image_ca, a_ca))
            r_inactive = float(cv_rmsd(image_ca, i_ca))
            if ph == PH_VALUES[0]:
                rmsd_records.append({"structure_id": sid, "f": f, "rmsd_to_active": r_active, "rmsd_to_inactive": r_inactive})

            print(f"    [{idx+1}/{N_IMAGES}] {sid} (f={f:.2f}): G = {g:.2f} kcal/mol, "
                  f"RMSD-to-active={r_active:.2f} A, RMSD-to-inactive={r_inactive:.2f} A "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)
        all_results[ph] = energies

    fractions = {sid: f for sid, (_, f) in image_paths.items()}
    table = pd.DataFrame({"structure_id": list(image_paths.keys()), "interpolation_fraction": [fractions[s] for s in image_paths]}).set_index("structure_id")
    for ph in PH_VALUES:
        table[f"gibbs_kcal_mol_ph{ph}"] = pd.Series(all_results[ph])
    rmsd_df = pd.DataFrame(rmsd_records).set_index("structure_id")
    table = table.join(rmsd_df[["rmsd_to_active", "rmsd_to_inactive"]])
    table.to_csv(out_dir / "path_table.csv")

    order = table.sort_values("interpolation_fraction")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ph, color in zip(PH_VALUES, ["#3b6fa0", "#c0533e"]):
        ax.plot(order["interpolation_fraction"], order[f"gibbs_kcal_mol_ph{ph}"], "o-", lw=2, color=color, label=f"pH {ph}")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("G (kcal/mol)")
    gate_note = "PASSES" if diagnostics["tm6_gate_4A_passed"] else "FAILS (not a genuine activation pair)"
    ax.set_title(f"{name}: G along corrected symmetric path (N={N_IMAGES})\n"
                 f"TM6 displacement={diagnostics['tm6_cytoplasmic_displacement_angstrom']:.2f} A -- {gate_note}", fontsize=9)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "gibbs_vs_path_both_ph.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ph, color in zip(PH_VALUES, ["#3b6fa0", "#c0533e"]):
        col = f"gibbs_kcal_mol_ph{ph}"
        rel = order[col] - order[col].iloc[0]
        ax.plot(order["interpolation_fraction"], rel, "o-", lw=2, color=color, label=f"pH {ph}")
    ax.axhline(0, color="#999999", lw=0.8, zorder=0)
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("G - G(active endpoint) (kcal/mol)")
    ax.set_title(f"{name}: barrier shape relative to active endpoint", fontsize=9)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "barrier_shape_relative.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.plot(order["interpolation_fraction"], order["rmsd_to_active"], "o-", label="RMSD to active", color="#3b6fa0")
    ax.plot(order["interpolation_fraction"], order["rmsd_to_inactive"], "o-", label="RMSD to inactive", color="#c0533e")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("Core CA RMSD (Å) after relaxation")
    ax.legend(frameon=False)
    ax.set_title(f"{name}: did relaxation preserve the interpolated path?", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "rmsd_sanity_check.png", dpi=200)
    plt.close(fig)

    summary = {"receptor": name, "diagnostics": diagnostics, "core_span": [core_lo, core_hi],
               "core_endpoint_rmsd_angstrom": core_endpoint_rmsd, "n_images": N_IMAGES,
               "ph_values": PH_VALUES, "minimize_iterations": MINIMIZE_ITERATIONS,
               "config_hash": sha256_of(__file__), "git_commit": git_commit_hash()}
    per_ph_summary = {}
    for ph in PH_VALUES:
        col = f"gibbs_kcal_mol_ph{ph}"
        peak_sid = order[col].idxmax()
        endpoint_active = float(order[col].iloc[0])
        endpoint_inactive = float(order[col].iloc[-1])
        lower_endpoint = min(endpoint_active, endpoint_inactive)
        barrier = float(order.loc[peak_sid, col]) - lower_endpoint
        per_ph_summary[str(ph)] = {
            "endpoint_active_G": endpoint_active,
            "endpoint_inactive_G": endpoint_inactive,
            "endpoint_shift_inactive_minus_active": endpoint_inactive - endpoint_active,
            "peak_f": float(order.loc[peak_sid, "interpolation_fraction"]),
            "peak_G": float(order.loc[peak_sid, col]),
            "barrier_vs_lower_endpoint_kcal_mol": barrier,
        }
    summary["per_ph"] = per_ph_summary
    summary["uniform_ph_offset_active_kcal_mol"] = per_ph_summary["6.0"]["endpoint_active_G"] - per_ph_summary["7.4"]["endpoint_active_G"]
    summary["barrier_change_ph74_to_ph60"] = per_ph_summary["6.0"]["barrier_vs_lower_endpoint_kcal_mol"] - per_ph_summary["7.4"]["barrier_vs_lower_endpoint_kcal_mol"]

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{name} DONE. Results in {out_dir}")
    print(f"  Core endpoint RMSD: {core_endpoint_rmsd:.2f} A")
    for ph in PH_VALUES:
        s = per_ph_summary[str(ph)]
        print(f"  pH {ph}: barrier={s['barrier_vs_lower_endpoint_kcal_mol']:.1f} kcal/mol at f={s['peak_f']:.2f}, "
              f"endpoint shift (inactive-active)={s['endpoint_shift_inactive_minus_active']:+.1f} kcal/mol")
    print(f"  Barrier change pH7.4->6.0: {summary['barrier_change_ph74_to_ph60']:+.1f} kcal/mol")
    print(f"  Uniform pH offset (active endpoint): {summary['uniform_ph_offset_active_kcal_mol']:+.1f} kcal/mol")

    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for name, paths in RECEPTORS.items():
        all_summaries[name] = run_one_receptor(name, paths["active"], paths["inactive"])

    with open(OUT_ROOT / "all_receptors_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    print(f"\n{'='*70}\nALL RECEPTORS DONE\n{'='*70}")
    for name, s in all_summaries.items():
        gate = "PASS" if s["diagnostics"]["tm6_gate_4A_passed"] else "FAIL"
        print(f"  {name}: TM6 disp={s['diagnostics']['tm6_cytoplasmic_displacement_angstrom']:.2f} A [{gate}], "
              f"barrier change pH7.4->6.0={s['barrier_change_ph74_to_ph60']:+.1f} kcal/mol")


if __name__ == "__main__":
    main()
