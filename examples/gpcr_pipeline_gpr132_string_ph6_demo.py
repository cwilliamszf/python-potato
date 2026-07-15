"""
Extends gpcr_pipeline_gpr132_string_demo.py's interpolated path with a
second pH (6.0), reusing the 11 already-built path structures (pure
Cartesian interpolation is geometrically pH-independent) rather than
rebuilding them -- mirrors gpcr_pipeline_gpr68_string_ph6_demo.py exactly.

Run with (~10-15 min, 11 real OpenMM minimizations at pH 6.0):

    PYTHONPATH=.:gibbs python examples/gpcr_pipeline_gpr132_string_ph6_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gibbs"))

from examples.gpcr_pipeline_gpr68_string_demo import run_tool3_gibbs  # noqa: E402
from examples.gpcr_pipeline_gpr132_string_demo import OUT_DIR, N_IMAGES  # noqa: E402

PH_NEW = 6.0
PH_EXISTING = 7.4


def main():
    table = pd.read_csv(OUT_DIR / "path_table.csv", index_col="structure_id")
    assert len(table) == N_IMAGES, f"expected {N_IMAGES} rows in path_table.csv, found {len(table)}"
    table = table.rename(columns={"gibbs_kcal_mol": f"gibbs_kcal_mol_ph{PH_EXISTING}"})

    path_dir = OUT_DIR / "path"
    energies_new = {}
    t0 = time.time()
    for i, sid in enumerate(table.index):
        pdb_path = path_dir / f"{sid}.pdb"
        assert pdb_path.exists(), f"missing structure file {pdb_path}"
        g = run_tool3_gibbs(pdb_path, PH_NEW)
        energies_new[sid] = g
        f = table.loc[sid, "interpolation_fraction"]
        print(f"  [{i+1}/{N_IMAGES}] {sid} (f={f:.2f}): G = {g:.2f} kcal/mol "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    table[f"gibbs_kcal_mol_ph{PH_NEW}"] = pd.Series(energies_new)
    table.to_csv(OUT_DIR / "path_table_both_ph.csv")

    order = table.sort_values("interpolation_fraction")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(order["interpolation_fraction"], order[f"gibbs_kcal_mol_ph{PH_EXISTING}"],
            "o-", lw=2, color="#3b6fa0", label=f"pH {PH_EXISTING}")
    ax.plot(order["interpolation_fraction"], order[f"gibbs_kcal_mol_ph{PH_NEW}"],
            "o-", lw=2, color="#c0533e", label=f"pH {PH_NEW}")
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("G (kcal/mol)")
    ax.set_title(f"GPR132: Gibbs energy along the interpolation path, pH {PH_EXISTING} vs. {PH_NEW}\n(N={N_IMAGES} images)", fontsize=9)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gibbs_vs_path_both_ph.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ph, color in [(PH_EXISTING, "#3b6fa0"), (PH_NEW, "#c0533e")]:
        col = f"gibbs_kcal_mol_ph{ph}"
        rel = order[col] - order[col].iloc[0]
        ax.plot(order["interpolation_fraction"], rel, "o-", lw=2, color=color, label=f"pH {ph}")
    ax.axhline(0, color="#999999", lw=0.8, zorder=0)
    ax.set_xlabel("Interpolation fraction (0=active, 1=inactive)")
    ax.set_ylabel("G - G(active endpoint) (kcal/mol)")
    ax.set_title("Barrier shape relative to the active endpoint, pH 7.4 vs. 6.0", fontsize=9)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "barrier_shape_relative.png", dpi=200)
    plt.close(fig)

    summary = {}
    for ph in (PH_EXISTING, PH_NEW):
        col = f"gibbs_kcal_mol_ph{ph}"
        peak_sid = order[col].idxmax()
        trough_sid = order[col].idxmin()
        barrier = order.loc[peak_sid, col] - min(order[col].iloc[0], order[col].iloc[-1])
        summary[str(ph)] = {
            "endpoint_active_G": float(order[col].iloc[0]),
            "endpoint_inactive_G": float(order[col].iloc[-1]),
            "peak_structure": peak_sid,
            "peak_f": float(order.loc[peak_sid, "interpolation_fraction"]),
            "peak_G": float(order.loc[peak_sid, col]),
            "trough_structure": trough_sid,
            "trough_f": float(order.loc[trough_sid, "interpolation_fraction"]),
            "trough_G": float(order.loc[trough_sid, col]),
            "barrier_vs_lower_endpoint_kcal_mol": float(barrier),
        }

    with open(OUT_DIR / "summary_both_ph.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Results in {OUT_DIR}")
    print(order[[f"gibbs_kcal_mol_ph{PH_EXISTING}", f"gibbs_kcal_mol_ph{PH_NEW}"]].to_string())
    for ph in (PH_EXISTING, PH_NEW):
        s = summary[str(ph)]
        print(f"  pH {ph}: barrier = {s['barrier_vs_lower_endpoint_kcal_mol']:.1f} kcal/mol "
              f"(peak at f={s['peak_f']:.2f}, trough at f={s['trough_f']:.2f})")
    endpoint_shift_active = summary[str(PH_NEW)]["endpoint_active_G"] - summary[str(PH_EXISTING)]["endpoint_active_G"]
    endpoint_shift_inactive = summary[str(PH_NEW)]["endpoint_inactive_G"] - summary[str(PH_EXISTING)]["endpoint_inactive_G"]
    peak_shift = summary[str(PH_NEW)]["peak_G"] - summary[str(PH_EXISTING)]["peak_G"]
    print(f"  pH6.0-pH7.4 shift: active endpoint {endpoint_shift_active:+.1f}, "
          f"inactive endpoint {endpoint_shift_inactive:+.1f}, peak/barrier region {peak_shift:+.1f} kcal/mol")


if __name__ == "__main__":
    main()
