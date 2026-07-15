import sys, json, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams
from wsme_gpcr.calibration import compute_fc

D = str(Path(__file__).resolve().parent)
STRUCT = f"{D}/structures"

# fixed xi per state: own pH-7 transition minus 3.0 J/mol (same matched-stability
# convention used throughout this session)
STATES = {
    "active":   (f"{STRUCT}/GPR68_active_core.pdb", None),
    "inactive": (f"{STRUCT}/GPR68_inactive_core.pdb", None),
}

PH_VALUES = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]

def main(fixed_xi):
    results = {}
    t_start = time.time()
    for tag, (path, _) in STATES.items():
        xi_j_mol = fixed_xi[tag]
        params = WSMEParams(T=310.0, ene=xi_j_mol * 1e-3)
        results[tag] = {"fixed_xi_j_mol": xi_j_mol, "ph": {}}
        for ph in PH_VALUES:
            t0 = time.time()
            r = run_pipeline(path, ph=ph, use_dssp=True, params=params, with_coupling=True)
            res = r.result
            cpl = r.coupling_result
            nb = r.block_model.nblocks
            amin = int(np.argmin(res.fes))
            fold_frac = float(res.n_values[amin] / nb)
            fc_pct = float(compute_fc(cpl, r.block_model) * 100)
            cfe = cpl.coupling_free_energy
            finite = np.isfinite(cfe)
            mean_abs_cpl = float(np.abs(cfe[finite]).mean())

            results[tag]["ph"][str(ph)] = {
                "nblocks": nb, "fold_frac": fold_frac, "fc_pct": fc_pct,
                "mean_abs_coupling_kj_mol": mean_abs_cpl,
                "fes": res.fes.tolist(), "n_values": res.n_values.tolist(),
                "fes2D": res.fes2D.tolist(), "hv": int(res.hv),
                "coupling_free_energy": cfe.tolist(),
            }
            print(f"[{tag} pH={ph}] nblocks={nb}, fold_frac={fold_frac:.1%}, fc={fc_pct:.1f}%, "
                  f"mean|cpl|={mean_abs_cpl:.2f} kJ/mol, t={time.time()-t0:.1f}s (cum={time.time()-t_start:.1f}s)", flush=True)
        with open(f"{D}/gpr68_ph_scan_results.json", "w") as f:
            json.dump(results, f, default=str)
    print(f"\nDONE, total {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--xi-active", type=float, required=True)
    p.add_argument("--xi-inactive", type=float, required=True)
    args = p.parse_args()
    main({"active": args.xi_active, "inactive": args.xi_inactive})
