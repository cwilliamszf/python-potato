import sys, json, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams
from wsme_gpcr.coupling import compute_coupling
from wsme_gpcr.calibration import compute_fc, DEFAULT_FC_Z_THRESHOLD
from wsme_gpcr.pka_predictor import predict_pka_propka
from wsme_gpcr.asr import parse_iqtree_state_file, site_to_resnum

S = str(Path(__file__).resolve().parent / "structures")
STATE_FILE = str(REPO_ROOT / "linkage_pka" / "asr_data" / "alignment_iqtree_asr_state.state")

NODES = {
    # tag: (core.cif path, iqtree node name, xi_transition_pH7_j_mol)
    "Node22": (f"{S}/node_22_core.cif", "Node22", -48.8999999999997),
    "Node21": (f"{S}/node_21_core.cif", "Node21", -51.49999999999974),
    "Node20": (f"{S}/node_20_core.cif", "Node20", -49.49999999999971),
    "Node80": (f"{S}/node_80_core.cif", "Node80", -50.89999999999973),
    "Node119": (f"{S}/node_119_core.cif", "Node119", -47.099999999999675),
    "Node32": (f"{S}/node_32_core.cif", "Node32", -50.89999999999973),
    "Node34": (f"{S}/node_34_core.cif", "Node34", -55.29999999999979),
    "Node70": (f"{S}/node_70_core.cif", "Node70", -54.49999999999978),
}

BW_SITES = {62: ("D2.50", "D", "triad"), 144: ("E4.53", "E", "triad"), 277: ("D7.49", "D", "triad"),
            79: ("H2.67", "H", "shell"), 164: ("H45.47", "H", "shell"), 264: ("H7.36", "H", "shell")}

PH_VALUES = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]
XI_OFFSETS = [-3.0, -5.0]
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(20260712)

posteriors = parse_iqtree_state_file(STATE_FILE)


def get_bw_resnums(iqnode):
    """Returns {label: (resnum_or_None, present_bool, actual_state)} for this node."""
    node = posteriors[iqnode]
    resnum_map = site_to_resnum(node)
    site_to_idx = {s: i for i, s in enumerate(node.site)}
    out = {}
    for site, (label, expected_aa, group) in BW_SITES.items():
        idx = site_to_idx[site]
        rn, state = int(resnum_map[idx]), node.state[idx]
        present = state == expected_aa
        out[label] = dict(resnum=rn if present else None, present=present, state=state, group=group)
    return out


def resnum_to_block(structure, block_model, resnum):
    matches = [i for i, rn in enumerate(structure.author_resnum) if rn == resnum]
    if not matches:
        return None
    return int(block_model.block_of_residue[matches[0]])


def block_centroids(structure, block_model):
    """CA-coordinate centroid per block, in the structure's own residue-index space."""
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


def triad_stat_and_null(cfe, group_blocks, exclude_blocks, dist_matrix, n_blocks, rng, n_bootstrap=N_BOOTSTRAP):
    """Mean |coupling| between group_blocks and all blocks not in group_blocks;
    plus a distance-matched null built by, for each (g, k) pair, substituting g
    with a random block r (excluding group_blocks and exclude_blocks and k
    itself) whose distance to k matches dist(g, k) within tolerance.

    Vectorized: the distance-matched candidate set for a given (g, k) pair is
    fixed (doesn't depend on the bootstrap draw), so it's computed once per
    pair, then all n_bootstrap draws for that pair are sampled from it in one
    vectorized numpy call rather than one Python-level draw at a time.
    """
    others = [k for k in range(n_blocks) if k not in group_blocks]
    if not others or not group_blocks:
        return None, None, None

    pairs = [(g, k) for g in group_blocks for k in others]
    observed_vals = np.array([abs(cfe[g, k]) for g, k in pairs])
    observed_mean = float(np.nanmean(observed_vals))

    candidate_pool = np.array([b for b in range(n_blocks) if b not in group_blocks and b not in exclude_blocks])

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
    return observed_mean, percentile, None


def run_condition(tag, path, iqnode, model, xi_j_mol, ph, bw_map):
    if model == "M":
        pka_overrides = None
    else:
        pka_overrides = predict_pka_propka(path)

    params = WSMEParams(T=310.0, ene=xi_j_mol * 1e-3)
    r = run_pipeline(path, ph=ph, use_dssp=True, pka_overrides=pka_overrides, params=params, with_coupling=True)
    st, bm = r.structure, r.block_model
    cpl = r.coupling_result
    cfe = cpl.coupling_free_energy

    triad_blocks, shell_blocks, absence = set(), set(), []
    for label, info in bw_map.items():
        if not info["present"]:
            absence.append(f"{label} absent (state={info['state']})")
            continue
        b = resnum_to_block(st, bm, info["resnum"])
        if b is None:
            absence.append(f"{label} resnum {info['resnum']} not found in truncated structure")
            continue
        (triad_blocks if info["group"] == "triad" else shell_blocks).add(b)

    centroids = block_centroids(st, bm)
    dist_matrix = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)

    exclude_for_triad_null = triad_blocks | shell_blocks
    exclude_for_shell_null = triad_blocks | shell_blocks

    triad_mean, triad_pct, _ = triad_stat_and_null(cfe, triad_blocks, exclude_for_triad_null, dist_matrix, bm.nblocks, RNG) \
        if triad_blocks else (None, None, None)
    shell_mean, shell_pct, _ = triad_stat_and_null(cfe, shell_blocks, exclude_for_shell_null, dist_matrix, bm.nblocks, RNG) \
        if shell_blocks else (None, None, None)

    # fc-style Z-score for triad/shell blocks specifically
    row_mean = np.nanmean(cfe, axis=1)
    valid = np.isfinite(row_mean)
    mu, sigma = row_mean[valid].mean(), row_mean[valid].std()
    z = (row_mean - mu) / sigma if sigma > 0 else np.zeros_like(row_mean)

    triad_z = float(np.mean([z[b] for b in triad_blocks])) if triad_blocks else None
    shell_z = float(np.mean([z[b] for b in shell_blocks])) if shell_blocks else None
    fc_pct = float(compute_fc(cpl, bm) * 100)

    return dict(
        nblocks=bm.nblocks, triad_blocks=sorted(triad_blocks), shell_blocks=sorted(shell_blocks),
        absence=absence, triad_mean_abs_cpl_kj_mol=triad_mean, triad_percentile_vs_null=triad_pct,
        shell_mean_abs_cpl_kj_mol=shell_mean, shell_percentile_vs_null=shell_pct,
        triad_mean_zscore=triad_z, shell_mean_zscore=shell_z, fc_pct=fc_pct,
    )


def main():
    results = {}
    t_start = time.time()
    for tag, (path, iqnode, xi_transition) in NODES.items():
        bw_map = get_bw_resnums(iqnode)
        results[tag] = {"bw_map": bw_map, "conditions": {}}
        for model in ["M", "P"]:
            for xi_offset in XI_OFFSETS:
                xi_j_mol = xi_transition + xi_offset
                for ph in PH_VALUES:
                    t0 = time.time()
                    key = f"{model}|xi{xi_offset}|pH{ph}"
                    out = run_condition(tag, path, iqnode, model, xi_j_mol, ph, bw_map)
                    results[tag]["conditions"][key] = out
                    print(f"[{tag} {key}] triad_mean={out['triad_mean_abs_cpl_kj_mol']}, "
                          f"triad_pct={out['triad_percentile_vs_null']}, shell_mean={out['shell_mean_abs_cpl_kj_mol']}, "
                          f"shell_pct={out['shell_percentile_vs_null']}, fc={out['fc_pct']:.1f}%, "
                          f"absence={out['absence']}, t={time.time()-t0:.1f}s (cum={time.time()-t_start:.1f}s)", flush=True)
        with open(str(Path(__file__).resolve().parent / "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nDONE, total {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
