"""
Real end-to-end 4-tool pipeline run on the actual GPR68 (OGR1) structure.

Unlike examples/gpcr_pipeline_real_demo.py (which used a mu-opioid receptor
structure as a stand-in because GPR68 wasn't reachable), this uses the real
GPR68 active- and inactive-state homology models the user provided:
examples/data/gpr68_structures/{active,inactive}/*.pdb (GPCRdb AFMS,
2024-05-15). GPR68 (OGR1) is a genuine proton-sensing class-A GPCR, so this
run is directly biologically relevant, not just a pipeline plumbing check.

What's real vs. substituted, same disclosure as the earlier demo:
  * Tool 1 (protonation from pKa + pH): REAL -- PROPKA3 structure-aware pKa
    on the actual GPR68 sequence/structure.
  * Tool 2 (AlphaFold/ColabFold ensemble): SUBSTITUTED with a Calpha-ANM
    normal-mode ensemble around the real GPR68 active-state model (no GPU/
    network available for actual ColabFold folding in this sandbox). Tool
    2's real activation-state code is still used on every conformer.
  * Tool 3 (Gibbs free energy): REAL -- gibbs/gpcr_gibbs_energy.py's actual
    AMBER ff14SB + GBn2 + RRHO pipeline via OpenMM/PDBFixer.
  * Tool 4 (landscape): REAL, unmodified.

GPR68's DRY-motif Arg and its TM7 P-x-x-Y motif (a "DPxxY" variant here,
not the canonical "NPxxY") are auto-detected from the real sequence rather
than assumed, since no external numbering database is reachable here either.

Run with (~25-35 min, N_CONFORMERS x 2 pH real OpenMM minimizations):

    PYTHONPATH=.:gibbs python examples/gpcr_pipeline_gpr68_demo.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, PDBParser
from Bio.SeqUtils import seq1

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gibbs"))

from gpcr_ensemble.activation_state import DEFAULT_THRESHOLDS, classify_model, tm3_tm6_distance  # noqa: E402
from gpcr_energy_landscapes import pipeline as landscape_pipeline  # noqa: E402
from gpcr_energy_landscapes import plotting  # noqa: E402
from gpcr_energy_landscapes.energy_landscape import landscape_1d  # noqa: E402
import gpcr_gibbs_energy as gibbs_tool  # noqa: E402
from wsme_gpcr.pka_predictor import predict_pka_propka  # noqa: E402

DATA_DIR = REPO_ROOT / "examples" / "data" / "gpr68_structures"
ACTIVE_PDB = DATA_DIR / "active" / "ClassA_ogr1_human_Active_AFMS_2024-05-15_GPCRdb.pdb"
INACTIVE_PDB = DATA_DIR / "inactive" / "ClassA_ogr1_human_Inactive_AFMS_2024-05-15_GPCRdb.pdb"
REFERENCE_PDB = ACTIVE_PDB  # conformers are generated around this state

OUT_DIR = Path(__file__).resolve().parent / "output" / "gpr68_demo"
N_CONFORMERS = 10
PH_VALUES = [7.4, 6.0]
RNG = np.random.default_rng(68)

FREE_SOLUTION_PKA = {"ASP": 3.9, "GLU": 4.1, "HIS": 6.0, "LYS": 10.5, "ARG": 12.5, "CYS": 8.3, "TYR": 10.1}
_ACID_RESNAMES = {"ASP", "GLU", "CYS", "TYR"}


def frac_protonated(pka_value, ph, acid=True):
    if acid:
        return 1.0 / (1.0 + 10 ** (ph - pka_value))
    return 1.0 / (1.0 + 10 ** (pka_value - ph))


def run_tool1_protonation(structure, pdb_path, chain_id="A"):
    print("[tool 1] Running PROPKA3 structure-aware pKa prediction on GPR68...", flush=True)
    pka = predict_pka_propka(pdb_path, chain=chain_id)
    print(f"[tool 1] Got pKa for {len(pka)} titratable groups.")

    resname_by_num = {r.id[1]: r.get_resname() for r in structure[0][chain_id] if r.id[0] == " "}
    shifts = []
    for resnum, pka_value in pka.items():
        resname = resname_by_num.get(resnum)
        baseline = FREE_SOLUTION_PKA.get(resname)
        if baseline is None:
            continue
        shifts.append((resnum, resname, pka_value, pka_value - baseline))
    shifts.sort(key=lambda t: -abs(t[3]))

    print("[tool 1] Largest structure-induced pKa shifts from free-solution baseline (top 10):")
    for resnum, resname, pka_value, shift in shifts[:10]:
        acid = resname in _ACID_RESNAMES
        f74 = frac_protonated(pka_value, 7.4, acid=acid)
        f60 = frac_protonated(pka_value, 6.0, acid=acid)
        print(
            f"    {resname}{resnum}: predicted pKa={pka_value:.2f} "
            f"(baseline {FREE_SOLUTION_PKA[resname]:.1f}, shift {shift:+.2f}); "
            f"protonated fraction pH7.4={f74:.2f} pH6.0={f60:.2f}"
        )
    return pka


def find_conserved_motifs(structure, chain_id="A"):
    """Auto-detect the DRY-motif Arg (TM3, 3.50) and the TM7 P-x-x-Y motif's
    Tyr (7.53) directly from sequence. GPR68 carries a DPxxY variant at the
    canonical NPxxY position (position 1 is D, not N) rather than the
    textbook-canonical NPxxY, so the position-1 residue is matched loosely
    ([NDS], the known class-A variants) rather than hardcoded to N."""
    chain = structure[0][chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    resnums = [r.id[1] for r in residues]
    seq = "".join(seq1(r.get_resname()) for r in residues)

    dry_idx = seq.find("DRY")
    if dry_idx == -1:
        dry_idx = seq.find("ERY")
    if dry_idx == -1:
        raise ValueError("Could not find a DRY/ERY motif in the sequence")
    arg_resnum = resnums[dry_idx + 1]

    npxxy_match = re.search("[NDS]P..Y", seq)
    if npxxy_match is None:
        # fall back to any P..Y in the C-terminal (TM7-containing) half
        candidates = list(re.finditer("P..Y", seq[len(seq) // 2 :]))
        if not candidates:
            raise ValueError("Could not find a P..Y (NPxxY-like) motif in the sequence")
        npxxy_match_start = candidates[-1].start() + len(seq) // 2
    else:
        npxxy_match_start = npxxy_match.start()
    tyr_resnum = resnums[npxxy_match_start + 4]

    return arg_resnum, tyr_resnum


def find_ionic_lock_partner(structure, arg_resnum, chain_id="A", min_seq_sep=15, max_dist=15.0, exclude_resnums=()):
    """Nearest Asp/Glu sidechain to the DRY-Arg's guanidinium group, excluding
    immediate sequence neighbors and any residue within min_seq_sep of an
    excluded reference position (e.g. the TM7 P-x-x-Y motif) -- otherwise the
    nearest acidic residue is liable to be part of that other motif itself
    (its own D/N position) rather than a genuine TM6 ionic-lock partner."""
    chain = structure[0][chain_id]
    arg_res = next(r for r in chain if r.id[1] == arg_resnum)
    guanidinium = np.array([a.coord for a in arg_res if a.get_name() in ("NH1", "NH2", "NE", "CZ")])

    best_resnum, best_dist = None, np.inf
    for res in chain:
        if res.id[0] != " " or res.get_resname() not in ("ASP", "GLU"):
            continue
        if abs(res.id[1] - arg_resnum) < min_seq_sep:
            continue
        if any(abs(res.id[1] - excl) < min_seq_sep for excl in exclude_resnums):
            continue
        coords = np.array([a.coord for a in res if a.element != "H"])
        d = np.linalg.norm(guanidinium[:, None, :] - coords[None, :, :], axis=-1).min()
        if d < best_dist:
            best_dist, best_resnum = d, res.id[1]
    if best_resnum is None or best_dist > max_dist:
        return None
    return best_resnum


def build_ca_anm_modes(ca_coords, cutoff_ang=15.0, gamma=1.0):
    n = len(ca_coords)
    hessian = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = ca_coords[i] - ca_coords[j]
            d = np.linalg.norm(d_vec)
            if d > cutoff_ang:
                continue
            block = -gamma * np.outer(d_vec, d_vec) / (d * d)
            hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
            hessian[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] += block
            hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= block
            hessian[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] -= block
    eigvals, eigvecs = np.linalg.eigh(hessian)
    return eigvals[6:], eigvecs[:, 6:]


def generate_anm_ensemble(structure, chain_id, n_conformers, out_dir, n_modes=10, target_rmsd_ang=0.8):
    chain = structure[0][chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    ca_coords = np.array([r["CA"].coord for r in residues])
    _, eigvecs = build_ca_anm_modes(ca_coords)

    out_dir.mkdir(parents=True, exist_ok=True)
    conformer_paths = {}
    writer = PDBIO()

    for k in range(n_conformers):
        amplitudes = RNG.normal(0, 1, n_modes)
        displacement = np.zeros_like(ca_coords)
        for m in range(n_modes):
            mode_vec = eigvecs[:, m].reshape(-1, 3)
            displacement += amplitudes[m] * mode_vec
        rmsd = np.sqrt((displacement**2).sum(axis=1).mean())
        displacement *= target_rmsd_ang / rmsd

        conf_structure = structure.copy()
        conf_chain = conf_structure[0][chain_id]
        conf_residues = [r for r in conf_chain if r.id[0] == " "]
        for res, shift in zip(conf_residues, displacement):
            for atom in res:
                atom.coord = atom.coord + shift

        sid = f"anm_{k:02d}"
        path = out_dir / f"{sid}.pdb"
        writer.set_structure(conf_structure)
        writer.save(str(path))
        conformer_paths[sid] = path

    return conformer_paths


def run_tool3_gibbs(pdb_path, ph):
    argv = [
        str(pdb_path),
        "--chains", "A",
        "--ph", str(ph),
        "--entropy-method", "ca-anm",
        "--minimize-iterations", "150",
        "--no-hbond-analysis",
    ]
    args = gibbs_tool.parse_args(argv)
    result = gibbs_tool.run(args)
    return result["G_kcal"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("gpr68_active", str(REFERENCE_PDB))
    inactive_ref = parser.get_structure("gpr68_inactive", str(INACTIVE_PDB))

    arg_resnum, tyr_resnum = find_conserved_motifs(structure)
    partner_resnum = find_ionic_lock_partner(structure, arg_resnum, exclude_resnums=(tyr_resnum,))
    print(f"GPR68 auto-detected DRY-motif Arg (TM3, 3.50): resnum {arg_resnum}")
    print(f"GPR68 auto-detected TM7 P-x-x-Y motif Tyr (7.53): resnum {tyr_resnum}")
    print(f"GPR68 auto-detected ionic-lock partner (nearest Asp/Glu): resnum {partner_resnum}")

    # Sanity check: same reference residue numbers should exist in the inactive model too
    inactive_ca = {r.id[1]: r["CA"].coord for r in inactive_ref[0]["A"] if r.id[0] == " "}
    active_ca = {r.id[1]: r["CA"].coord for r in structure[0]["A"] if r.id[0] == " "}
    if partner_resnum:
        d_active = tm3_tm6_distance(active_ca, arg_resnum, partner_resnum)
        d_inactive = tm3_tm6_distance(inactive_ca, arg_resnum, partner_resnum)
        print(f"Reference ionic-lock distance: active model={d_active:.2f} A, inactive model={d_inactive:.2f} A")

    pka = run_tool1_protonation(structure, REFERENCE_PDB)

    print(f"\n[tool 2-substitute] Generating {N_CONFORMERS} ANM-displaced GPR68 conformers...")
    ensemble_dir = OUT_DIR / "ensemble"
    conformer_paths = generate_anm_ensemble(structure, "A", N_CONFORMERS, ensemble_dir)

    cv1_by_sid = {}
    cv2_by_sid = {}
    state_by_sid = {}
    for sid, path in conformer_paths.items():
        conf = parser.get_structure(sid, str(path))
        chain = conf[0]["A"]
        ca_coords = {r.id[1]: r["CA"].coord for r in chain if r.id[0] == " "}
        d_lock = tm3_tm6_distance(ca_coords, arg_resnum, partner_resnum) if partner_resnum else np.nan
        cv1_by_sid[sid] = d_lock
        tyr_res = next(r for r in chain if r.id[1] == tyr_resnum)
        arg_res = next(r for r in chain if r.id[1] == arg_resnum)
        cv2_by_sid[sid] = float(np.linalg.norm(tyr_res["CA"].coord - arg_res["CA"].coord))
        label, _ = classify_model(ca_coords, arg_resnum, partner_resnum, DEFAULT_THRESHOLDS) if partner_resnum else ("n/a", None)
        state_by_sid[sid] = label

    print(f"[tool 2-substitute] Activation-state labels: {state_by_sid}")

    all_results = {}
    for ph in PH_VALUES:
        print(f"\n[tool 3] Running real Gibbs free-energy calculations on GPR68 at pH {ph} "
              f"({N_CONFORMERS} conformers)...")
        energies = {}
        t0 = time.time()
        for i, (sid, path) in enumerate(conformer_paths.items()):
            g = run_tool3_gibbs(path, ph)
            energies[sid] = g
            print(f"  [{i+1}/{N_CONFORMERS}] {sid}: G = {g:.2f} kcal/mol "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)
        all_results[ph] = energies

        energies_df = pd.DataFrame(
            {"structure_id": list(energies.keys()), "gibbs_kcal_mol": list(energies.values())}
        ).set_index("structure_id")
        cv_table = pd.DataFrame(
            {
                "structure_id": list(conformer_paths.keys()),
                "ionic_lock_like": [cv1_by_sid[s] for s in conformer_paths],
                "pxxy_dry_distance": [cv2_by_sid[s] for s in conformer_paths],
            }
        ).set_index("structure_id")
        merged = landscape_pipeline.merge_with_energies(cv_table, energies_df)
        merged.to_csv(OUT_DIR / f"merged_ph{ph}.csv")

        landscape1d = landscape_pipeline.build_1d_landscape(merged, "ionic_lock_like", method="kde", grid_size=100)
        fig, ax = plt.subplots(figsize=(5, 4))
        plotting.plot_1d_landscape(landscape1d, cv_label="DRY-Arg <-> ionic-lock-partner distance (Å)", ax=ax)
        ax.set_title(f"GPR68, pH {ph} (real Gibbs energies, N={N_CONFORMERS})", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"landscape_1d_ph{ph}.png", dpi=200)
        plt.close(fig)

        landscape2d = landscape_pipeline.build_2d_landscape(
            merged, "ionic_lock_like", "pxxy_dry_distance", method="kde", grid_size=80
        )
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plotting.plot_2d_landscape(
            landscape2d,
            x_label="DRY-Arg <-> ionic-lock-partner distance (Å)",
            y_label="Tyr(TM7) <-> DRY-Arg distance (Å)",
            ax=ax,
            scatter=merged,
            scatter_x="ionic_lock_like",
            scatter_y="pxxy_dry_distance",
        )
        ax.set_title(f"GPR68, pH {ph} (N={N_CONFORMERS})", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"landscape_2d_ph{ph}.png", dpi=200)
        plt.close(fig)

    # Overlay absolute G vs pH (the per-pH landscapes are individually
    # renormalized to their own minimum, which hides any absolute shift --
    # see the earlier demo's README for why this companion plot matters)
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    y74 = [all_results[7.4][s] for s in conformer_paths]
    y60 = [all_results[6.0][s] for s in conformer_paths]
    for a, b in zip(y74, y60):
        ax.plot([0, 1], [a, b], color="#999999", lw=1, zorder=1)
    ax.scatter(np.zeros(len(y74)), y74, s=50, color="#3b6fa0", zorder=2, label="pH 7.4")
    ax.scatter(np.ones(len(y60)), y60, s=50, color="#c0533e", zorder=2, label="pH 6.0")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pH 7.4", "pH 6.0"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("G (kcal/mol)")
    ax.set_title("GPR68: real AMBER ff14SB + GBn2 + RRHO Gibbs energy vs. pH\n"
                  f"(N={N_CONFORMERS} ANM conformers)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gibbs_vs_ph.png", dpi=200)
    plt.close(fig)

    # 1D landscape overlay across pH (real, not per-pH renormalized shape trick
    # -- shown for completeness even though CVs are pH-independent by construction)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for ph, color in zip(PH_VALUES, ["#3b6fa0", "#c0533e"]):
        cv = np.array([cv1_by_sid[s] for s in conformer_paths])
        g = np.array([all_results[ph][s] for s in conformer_paths])
        ld = landscape_1d(cv, gibbs=g, method="kde", grid_size=100)
        ax.plot(ld["cv"], ld["dG"], lw=2, color=color, label=f"pH {ph}")
    ax.set_xlabel("DRY-Arg <-> ionic-lock-partner distance (Å)")
    ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "landscape_1d_ph_overlay.png", dpi=200)
    plt.close(fig)

    summary = {
        "reference_structure": str(REFERENCE_PDB),
        "inactive_reference_structure": str(INACTIVE_PDB),
        "dry_arg_resnum": arg_resnum,
        "pxxy_tyr_resnum": tyr_resnum,
        "ionic_lock_partner_resnum": partner_resnum,
        "reference_ionic_lock_distance_active": float(d_active) if partner_resnum else None,
        "reference_ionic_lock_distance_inactive": float(d_inactive) if partner_resnum else None,
        "activation_states": state_by_sid,
        "pka_sample": dict(list(sorted(pka.items()))[:30]),
        "gibbs_kcal_mol_by_ph": all_results,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Results in {OUT_DIR}")
    for ph, energies in all_results.items():
        vals = list(energies.values())
        print(f"  pH {ph}: G range [{min(vals):.1f}, {max(vals):.1f}] kcal/mol, mean {np.mean(vals):.1f}")
    mean_shift = np.mean(y60) - np.mean(y74)
    print(f"  mean G shift (pH6.0 - pH7.4): {mean_shift:+.1f} kcal/mol")


if __name__ == "__main__":
    main()
