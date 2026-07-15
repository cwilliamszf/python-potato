import sys
sys.path.insert(0, "/home/user/python-potato")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams
from wsme_gpcr.plotting import plot_comparison_grid, plot_coupling_matrix, plot_2d_landscape

D = "/tmp/claude-0/-home-user-python-potato/e6c23a7d-0f3f-50fe-a92b-cd58fe8f9e63/scratchpad/gpcrdb_downloads"
PH_VALUES = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]
FIXED_XI = {"active": -58.1, "inactive": -54.3}
PATHS = {"active": f"{D}/GPR68_active_core.pdb", "inactive": f"{D}/GPR68_inactive_core.pdb"}

for tag in ["active", "inactive"]:
    params = WSMEParams(T=310.0, ene=FIXED_XI[tag] * 1e-3)
    results_by_key, coupling_by_key = {}, {}
    for ph in PH_VALUES:
        r = run_pipeline(PATHS[tag], ph=ph, use_dssp=True, params=params, with_coupling=True)
        key = f"pH {ph}"
        results_by_key[key] = r.result
        coupling_by_key[key] = r.coupling_result
        print(f"{tag} pH={ph} done", flush=True)

    fig = plot_comparison_grid(results_by_key, coupling_by_key, figsize_per_panel=4.2)
    fig.suptitle(f"GPR68 {tag} (fixed xi = {FIXED_XI[tag]} J/mol, DSSP blocking)", fontsize=14, y=1.01)
    fig.savefig(f"{D}/GPR68_{tag}_comparison_grid.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved GPR68_{tag}_comparison_grid.png", flush=True)

print("ALL DONE", flush=True)
