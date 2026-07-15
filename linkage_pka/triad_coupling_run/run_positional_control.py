import sys, json, time, traceback
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams
from wsme_gpcr.coupling import compute_coupling
from wsme_gpcr.calibration import compute_fc
from wsme_gpcr.pka_predictor import predict_pka_propka
from wsme_gpcr.asr import parse_iqtree_state_file, site_to_resnum
from wsme_gpcr.structure import CHARGED_RESIDUES, DEFAULT_PKA, fraction_charged

S = str(Path(__file__).resolve().parent / "structures")
STATE_FILE = str(REPO_ROOT / "linkage_pka" / "asr_data" / "alignment_iqtree_asr_state.state")
GPR68_PDB = f"{S}/gpr68_prep/inactive_prepped.pdb"

# tag: (path, iqtree_node_or_None, xi_transition_pH7_j_mol, lineage)
NODES = {
    "Node22":  (f"{S}/node_22_core.cif",  "Node22",  -48.8999999999997,   "sensor"),
    "Node21":  (f"{S}/node_21_core.cif",  "Node21",  -51.49999999999974,  "sensor"),
    "Node20":  (f"{S}/node_20_core.cif",  "Node20",  -49.49999999999971,  "sensor"),
    "Node80":  (f"{S}/node_80_core.cif",  "Node80",  -50.89999999999973,  "sensor"),
    "Node119": (f"{S}/node_119_core.cif", "Node119", -47.099999999999675, "sensor"),
    "Node32":  (f"{S}/node_32_core.cif",  "Node32",  -50.89999999999973,  "non_sensor"),
    "Node34":  (f"{S}/node_34_core.cif",  "Node34",  -55.29999999999979,  "non_sensor"),
    "Node70":  (f"{S}/node_70_core.cif",  "Node70",  -54.49999999999978,  "non_sensor"),
}

# real GPR68: added separately below (best-effort; may fail to load given it was
# prepped for a different pipeline -- if so, excluded and reported, not faked)
GPR68_XI_TRANSITION_PH7_J_MOL = None  # resolved via a quick xi_fold_scan before the main loop, see main()

BW_POSITIONS = {"pos2.50": 62, "pos4.53": 144, "pos7.49": 277}
SHELL_SITES = {"H2.67": 79, "H45.47": 164, "H7.36": 264}

PH_VALUES = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]
XI_OFFSETS = [-3.0, -5.0]
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(20260713)

posteriors = parse_iqtree_state_file(STATE_FILE)

# real GPR68 resnum mapping, from the manuscript's own published residue table
# (Figure 2 Panel B: hs.GPR68 column) -- D2.50=Asp67, E4.53=Glu149, D7.49=Asp282.
# Histidine-shell (H2.67/H45.47/H7.36) resnums for real GPR68 were NOT independently
# confirmed against that table this session -- omitted rather than guessed.
GPR68_POSITION_RESNUM = {"pos2.50": 67, "pos4.53": 149, "pos7.49": 282}


def get_bw_positions_asr(iqnode):
    """Returns {pos_label: dict(resnum, state)} for an ASR node -- ALWAYS resolves
    a resnum for every position regardless of residue identity (the positional-
    control fix: identity is reported, not used to gate inclusion)."""
    node = posteriors[iqnode]
    resnum_map = site_to_resnum(node)
    site_to_idx = {s: i for i, s in enumerate(node.site)}
    out = {}
    for label, site in BW_POSITIONS.items():
        idx = site_to_idx[site]
        out[label] = dict(resnum=int(resnum_map[idx]), state=node.state[idx])
    return out


def get_shell_positions_asr(iqnode):
    node = posteriors[iqnode]
    resnum_map = site_to_resnum(node)
    site_to_idx = {s: i for i, s in enumerate(node.site)}
    out = {}
    for label, site in SHELL_SITES.items():
        idx = site_to_idx[site]
        out[label] = dict(resnum=int(resnum_map[idx]), state=node.state[idx])
    return out


def resnum_to_block(structure, block_model, resnum):
    matches = [i for i, rn in enumerate(structure.author_resnum) if rn == resnum]
    if not matches:
        return None
    return int(block_model.block_of_residue[matches[0]])


def block_centroids(structure, block_model):
    ca_coord = {}
    for i, name in enumerate(structure.atom_name):
        if name == "CA":
            ridx = int(structure.atom_resindex[i])
            ca_coord[ridx] = structure.coord[i]
    centroids = np.zeros((block_model.nblocks, 3))
    for b in range(block_model.nblocks):
        lo, hi = block_model.block_residue_range[b]
        pts = [ca_coord[r] for r in range(lo, hi + 1) if r in ca_coord]
        centroids[b] = np.mean(pts, axis=0) if pts else np.nan
    return centroids


def group_stat_and_null(cfe, group_blocks, exclude_blocks, dist_matrix, n_blocks, rng, n_bootstrap=N_BOOTSTRAP):
    others = [k for k in range(n_blocks) if k not in group_blocks]
    if not others or not group_blocks:
        return None, None

    pairs = [(g, k) for g in group_blocks for k in others]
    observed_vals = np.array([abs(cfe[g, k]) for g, k in pairs])
    observed_mean = float(np.nanmean(observed_vals))

    candidate_pool = np.array([b for b in range(n_blocks) if b not in group_blocks and b not in exclude_blocks])
    if len(candidate_pool) == 0:
        return observed_mean, None

    draws = np.empty((len(pairs), n_bootstrap))
    for i, (g, k) in enumerate(pairs):
        d_gk = dist_matrix[g, k]
        tol = max(2.0, 0.15 * d_gk)
        mask = (candidate_pool != k) & (np.abs(dist_matrix[candidate_pool, k] - d_gk) <= tol)
        tries = 0
        while not mask.any() and tries < 5:
            tol *= 1.5
            mask = (candidate_pool != k) & (np.abs(dist_matrix[candidate_pool, k] - d_gk) <= tol)
            tries += 1
        cands = candidate_pool[mask] if mask.any() else candidate_pool[candidate_pool != k]
        if len(cands) == 0:
            cands = candidate_pool
        vals = np.abs(cfe[cands, k])
        idx = rng.integers(0, len(cands), size=n_bootstrap)
        draws[i] = vals[idx]

    null_means = np.nanmean(draws, axis=0)
    percentile = float(100.0 * (np.sum(null_means <= observed_mean) + 1) / (n_bootstrap + 1))
    return observed_mean, percentile


def titration_info(resname, resnum, ph_list, model, propka_overrides):
    """Charged-at-pH7 and titratable-in-8-to-5 for a single position, under the
    given charge model. Non-charged residue types (V, N, Q, S, T, ...) are
    trivially non-titratable -- reported explicitly, not silently skipped."""
    if resname not in CHARGED_RESIDUES:
        return dict(charged_type=False, pka_used=None, charged_frac_ph7=0.0,
                     charged_frac_ph8=0.0, charged_frac_ph5=0.0, titratable_8to5=False,
                     note=f"residue type {resname} is not a titratable group in this model")
    pka = DEFAULT_PKA[resname] if model == "M" else propka_overrides.get(resnum, DEFAULT_PKA[resname])
    f7 = fraction_charged(7.0, pka, resname)
    f8 = fraction_charged(8.0, pka, resname)
    f5 = fraction_charged(5.0, pka, resname)
    return dict(charged_type=True, pka_used=pka, charged_frac_ph7=f7, charged_frac_ph8=f8,
                charged_frac_ph5=f5, titratable_8to5=bool(abs(f8 - f5) > 0.1), note=None)


def run_condition(path, model, xi_j_mol, ph, position_map, shell_map):
    pka_overrides = None if model == "M" else predict_pka_propka(path)
    params = WSMEParams(T=310.0, ene=xi_j_mol * 1e-3)
    r = run_pipeline(path, ph=ph, use_dssp=True, pka_overrides=pka_overrides, params=params, with_coupling=True)
    st, bm = r.structure, r.block_model
    cpl = r.coupling_result
    cfe = cpl.coupling_free_energy

    def resname_at(resnum):
        matches = [rn for rn2, rn in zip(st.author_resnum, st.resname) if rn2 == resnum]
        return matches[0] if matches else None

    pos_blocks, pos_info = set(), {}
    for label, info in position_map.items():
        b = resnum_to_block(st, bm, info["resnum"])
        rname = resname_at(info["resnum"])
        pos_info[label] = dict(resnum=info["resnum"], expected_state=info["state"], resname_in_structure=rname,
                                block=b, **titration_info(rname, info["resnum"], PH_VALUES, model, pka_overrides or {}))
        if b is not None:
            pos_blocks.add(b)

    shell_blocks = set()
    if shell_map:
        for label, info in shell_map.items():
            b = resnum_to_block(st, bm, info["resnum"])
            if b is not None and info["state"] == "H":
                shell_blocks.add(b)

    centroids = block_centroids(st, bm)
    dist_matrix = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    exclude = pos_blocks | shell_blocks

    pos_mean, pos_pct = group_stat_and_null(cfe, pos_blocks, exclude, dist_matrix, bm.nblocks, RNG)
    shell_mean, shell_pct = (group_stat_and_null(cfe, shell_blocks, exclude, dist_matrix, bm.nblocks, RNG)
                              if shell_blocks else (None, None))

    fc_pct = float(compute_fc(cpl, bm) * 100)

    return dict(nblocks=bm.nblocks, position_blocks=sorted(pos_blocks), shell_blocks=sorted(shell_blocks),
                position_info=pos_info, positional_mean_abs_cpl_kj_mol=pos_mean,
                positional_percentile_vs_null=pos_pct, shell_mean_abs_cpl_kj_mol=shell_mean,
                shell_percentile_vs_null=shell_pct, fc_pct=fc_pct)


def resolve_gpr68_xi_transition():
    """Locate GPR68's own pH-7 fold/unfold transition via xi_fold_scan, exactly
    as done for every ASR node earlier this session, so the same
    transition-minus-{3,5} convention applies. Returns None (with a message) if
    the real prepped GPR68 structure doesn't load cleanly in this pipeline."""
    from wsme_gpcr.calibration import xi_fold_scan
    try:
        r = run_pipeline(GPR68_PDB, ph=7.0, use_dssp=True)
        scan = xi_fold_scan(r.structure, r.block_model, r.ss_mask, WSMEParams(T=310.0),
                             xi_range_j_mol=(-70.0, -38.0), step_j_mol=0.2)
        if not scan.folds_anywhere or scan.n_transitions != 1:
            return None, f"folds_anywhere={scan.folds_anywhere}, n_transitions={scan.n_transitions} -- not a clean single transition"
        fold_ok_mask = scan.fold_fracs >= 0.85
        for i in range(1, len(fold_ok_mask)):
            if fold_ok_mask[i] != fold_ok_mask[i - 1]:
                return float((scan.xi_values_j_mol[i - 1] + scan.xi_values_j_mol[i]) / 2.0), None
        return None, "no transition crossing found"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    results = {"nodes": {}, "gpr68": None, "gpr68_error": None,
               "unavailable": ["extant GPR4", "extant GPR65", "extant GPR132", "extant GPR184",
                                "Node62 (actinopterygian GPR132)"]}
    t_start = time.time()

    print("Resolving real GPR68's own xi transition...", flush=True)
    gpr68_xi, gpr68_err = resolve_gpr68_xi_transition()
    if gpr68_xi is None:
        print(f"  GPR68 EXCLUDED: {gpr68_err}", flush=True)
        results["gpr68_error"] = gpr68_err
    else:
        print(f"  GPR68 transition = {gpr68_xi:.2f} J/mol, proceeding", flush=True)
        NODES["GPR68"] = (GPR68_PDB, None, gpr68_xi, "sensor")

    for tag, (path, iqnode, xi_transition, lineage) in NODES.items():
        try:
            if tag == "GPR68":
                position_map = {k: dict(resnum=v, state=None) for k, v in GPR68_POSITION_RESNUM.items()}
                shell_map = None  # not independently verified for real GPR68 -- omitted
            else:
                position_map = get_bw_positions_asr(iqnode)
                shell_map = get_shell_positions_asr(iqnode)
        except Exception:
            print(f"[{tag}] FAILED to resolve positions: {traceback.format_exc()}", flush=True)
            continue

        results["nodes"][tag] = {"lineage": lineage, "position_map": position_map,
                                  "shell_map": shell_map, "conditions": {}}
        for model in ["M", "P"]:
            for xi_offset in XI_OFFSETS:
                xi_j_mol = xi_transition + xi_offset
                for ph in PH_VALUES:
                    t0 = time.time()
                    key = f"{model}|xi{xi_offset}|pH{ph}"
                    try:
                        out = run_condition(path, model, xi_j_mol, ph, position_map, shell_map)
                    except Exception:
                        print(f"[{tag} {key}] ERROR: {traceback.format_exc()}", flush=True)
                        continue
                    results["nodes"][tag]["conditions"][key] = out
                    print(f"[{tag} {key}] pos_mean={out['positional_mean_abs_cpl_kj_mol']}, "
                          f"pos_pct={out['positional_percentile_vs_null']}, shell_mean={out['shell_mean_abs_cpl_kj_mol']}, "
                          f"fc={out['fc_pct']:.1f}%, t={time.time()-t0:.1f}s (cum={time.time()-t_start:.1f}s)", flush=True)
        with open(str(Path(__file__).resolve().parent / "positional_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nDONE, total {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
