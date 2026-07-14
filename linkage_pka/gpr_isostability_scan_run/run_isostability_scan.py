import sys, json, time
sys.path.insert(0, "/home/user/python-potato")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams
from wsme_gpcr.calibration import (
    compute_delta_g_fold, calibrate_xi_isostability_mode, PAPER_XI_BRACKET_KJ_MOL, DEFAULT_FC_Z_THRESHOLD,
)
from wsme_gpcr.calibration import compute_fc
from wsme_gpcr.plotting import plot_comparison_grid

D68 = "/tmp/claude-0/-home-user-python-potato/e6c23a7d-0f3f-50fe-a92b-cd58fe8f9e63/scratchpad/gpcrdb_downloads"
D = "/tmp/claude-0/-home-user-python-potato/e6c23a7d-0f3f-50fe-a92b-cd58fe8f9e63/scratchpad/gpcrdb_downloads/gpr4_gpr65"

# (active_path, inactive_path, old fixed-xi for active [J/mol], used as the
# iso-stability REFERENCE for that receptor's pair -- unchanged from the
# previous ad hoc run)
RECEPTORS = {
    "gpr4":   (f"{D}/gpr4_active_core.pdb",   f"{D}/gpr4_inactive_core.pdb",   -53.5),
    "gpr65":  (f"{D}/gpr65_active_core.pdb",  f"{D}/gpr65_inactive_core.pdb",  -49.3),
    "gpr68":  (f"{D68}/GPR68_active_core.pdb", f"{D68}/GPR68_inactive_core.pdb", -58.1),
    "gpr132": (f"{D}/gpr132_active_core.pdb", f"{D}/gpr132_inactive_core.pdb", -58.1),
}

PH_VALUES = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]
T_REF = 310.0

calib_summary = {}
xi_by_tag = {}

print("=== ISO-STABILITY CALIBRATION (active = reference, per receptor) ===", flush=True)
for recep, (act_path, inact_path, xi_active_j_mol) in RECEPTORS.items():
    t0 = time.time()
    r_act = run_pipeline(act_path, ph=7.0, use_dssp=True, params=WSMEParams(T=T_REF))
    r_inact = run_pipeline(inact_path, ph=7.0, use_dssp=True, params=WSMEParams(T=T_REF))

    xi_ref_kj = xi_active_j_mol * 1e-3
    iso = calibrate_xi_isostability_mode(
        reference_structure=r_act.structure, reference_block_model=r_act.block_model, reference_ss_mask=r_act.ss_mask,
        xi_reference_kj_mol=xi_ref_kj,
        other_structure=r_inact.structure, other_block_model=r_inact.block_model, other_ss_mask=r_inact.ss_mask,
        params=WSMEParams(T=T_REF),
        xi_bracket_kj_mol=PAPER_XI_BRACKET_KJ_MOL,
        reference_structure_path=act_path, other_structure_path=inact_path,
    )
    xi_by_tag[f"{recep}_active"] = xi_active_j_mol
    xi_by_tag[f"{recep}_inactive"] = iso.xi_other_kj_mol * 1e3

    calib_summary[recep] = dict(
        xi_active_j_mol=xi_active_j_mol,
        xi_inactive_j_mol_OLD_adhoc=None,  # filled in below from prior run for comparison
        xi_inactive_j_mol_isostability=iso.xi_other_kj_mol * 1e3,
        delta_g_fold_common_kj_mol=iso.delta_g_fold_common_kj_mol,
        warning=iso.warning,
    )
    print(f"[{recep}] xi_active={xi_active_j_mol:.2f} J/mol (reference), "
          f"xi_inactive(iso-stability)={iso.xi_other_kj_mol*1e3:.2f} J/mol, "
          f"common dG_fold={iso.delta_g_fold_common_kj_mol:.2f} kJ/mol at {T_REF:.0f}K, "
          f"t={time.time()-t0:.1f}s", flush=True)

with open(f"{D}/isostability_calibration.json", "w") as f:
    json.dump(calib_summary, f, indent=2)

# old ad hoc xi (from the previous, non-iso-stability run) for direct comparison
OLD_XI_J_MOL = {
    "gpr4_active": -53.5, "gpr4_inactive": -53.7,
    "gpr65_active": -49.3, "gpr65_inactive": -57.1,
    "gpr68_active": -58.1, "gpr68_inactive": -54.3,
    "gpr132_active": -58.1, "gpr132_inactive": -68.7,
}

print("\n=== FULL pH SCAN AT ISO-STABILITY-CALIBRATED xi ===", flush=True)
summary = {}
t_start = time.time()
paths = {}
for recep, (act_path, inact_path, _) in RECEPTORS.items():
    paths[f"{recep}_active"] = act_path
    paths[f"{recep}_inactive"] = inact_path

for tag, path in paths.items():
    xi_j_mol = xi_by_tag[tag]
    params = WSMEParams(T=T_REF, ene=xi_j_mol * 1e-3)
    summary[tag] = {"fixed_xi_j_mol": xi_j_mol, "old_adhoc_xi_j_mol": OLD_XI_J_MOL[tag], "ph": {}}
    results_by_key, coupling_by_key = {}, {}
    for ph in PH_VALUES:
        t0 = time.time()
        r = run_pipeline(path, ph=ph, use_dssp=True, params=params, with_coupling=True)
        res, cpl, nb = r.result, r.coupling_result, r.block_model.nblocks
        amin = int(np.argmin(res.fes))
        fold_frac = float(res.n_values[amin] / nb)
        fc_pct = float(compute_fc(cpl, r.block_model) * 100)
        cfe = cpl.coupling_free_energy
        finite = np.isfinite(cfe)
        mean_abs_cpl = float(np.abs(cfe[finite]).mean())
        summary[tag]["ph"][str(ph)] = dict(nblocks=nb, fold_frac=fold_frac, fc_pct=fc_pct, mean_abs_coupling_kj_mol=mean_abs_cpl)
        results_by_key[f"pH {ph}"] = res
        coupling_by_key[f"pH {ph}"] = cpl
        print(f"[{tag} pH={ph}] fold_frac={fold_frac:.1%}, fc={fc_pct:.1f}%, mean|cpl|={mean_abs_cpl:.2f} kJ/mol, "
              f"t={time.time()-t0:.1f}s (cum={time.time()-t_start:.1f}s)", flush=True)

    fig = plot_comparison_grid(results_by_key, coupling_by_key, figsize_per_panel=4.2)
    fig.suptitle(f"{tag} (iso-stability xi = {xi_j_mol:.2f} J/mol, DSSP blocking)", fontsize=14, y=1.01)
    fig.savefig(f"{D}/{tag}_isostability_comparison_grid.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {tag}_isostability_comparison_grid.png", flush=True)

    with open(f"{D}/isostability_ph_scan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

print(f"\nALL DONE, total {time.time()-t_start:.1f}s", flush=True)
