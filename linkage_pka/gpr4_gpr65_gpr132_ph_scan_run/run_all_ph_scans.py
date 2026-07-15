import sys
sys.path.insert(0, "/home/user/python-potato")
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams
from wsme_gpcr.calibration import compute_fc
from wsme_gpcr.plotting import plot_comparison_grid

D = "/tmp/claude-0/-home-user-python-potato/e6c23a7d-0f3f-50fe-a92b-cd58fe8f9e63/scratchpad/gpcrdb_downloads/gpr4_gpr65"
PH_VALUES = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]

# transition (pH7) minus 3.0 J/mol, same matched-stability convention as GPR68
STATES = {
    "gpr4_active":    (f"{D}/gpr4_active_core.pdb",   -50.5 - 3.0),
    "gpr4_inactive":  (f"{D}/gpr4_inactive_core.pdb", -50.7 - 3.0),
    "gpr65_active":   (f"{D}/gpr65_active_core.pdb",  -46.3 - 3.0),
    "gpr65_inactive": (f"{D}/gpr65_inactive_core.pdb",-54.1 - 3.0),
    "gpr132_active":   (f"{D}/gpr132_active_core.pdb",  -55.1 - 3.0),
    "gpr132_inactive": (f"{D}/gpr132_inactive_core.pdb",-65.7 - 3.0),
}

summary = {}
t_start = time.time()
for tag, (path, xi_j_mol) in STATES.items():
    params = WSMEParams(T=310.0, ene=xi_j_mol * 1e-3)
    summary[tag] = {"fixed_xi_j_mol": xi_j_mol, "ph": {}}
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
    fig.suptitle(f"{tag} (fixed xi = {xi_j_mol:.1f} J/mol, DSSP blocking)", fontsize=14, y=1.01)
    fig.savefig(f"{D}/{tag}_comparison_grid.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {tag}_comparison_grid.png", flush=True)

    with open(f"{D}/all_ph_scan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

print(f"\nALL DONE, total {time.time()-t_start:.1f}s", flush=True)
