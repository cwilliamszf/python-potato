import numpy as np
from pathlib import Path

from wsme_gpcr.alanine_scan import (
    alanine_exclude_mask,
    estimate_scan_seconds,
    pca_cluster_residues,
    ph_cluster_table,
    ph_sensitivity_table,
    residue_ph_features,
    run_alanine_scan,
    scannable_positions,
    subsample_positions,
)
from wsme_gpcr.blocking import build_blocks
from wsme_gpcr.contacts import compute_contact_map
from wsme_gpcr.secondary_structure import assign_secondary_structure
from wsme_gpcr.structure import load_structure
from wsme_gpcr.wsme import WSMEParams

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"


def test_scannable_positions_excludes_ala_gly_pro():
    s = load_structure(CI2)
    scanned = set(scannable_positions(s))
    for i, rn in enumerate(s.resname):
        resnum = int(s.author_resnum[i])
        if rn in ("ALA", "GLY", "PRO"):
            assert resnum not in scanned
        else:
            assert resnum in scanned


def test_alanine_mask_is_noop_for_existing_alanine():
    s = load_structure(CI2)
    ala_resnum = next(int(s.author_resnum[i]) for i, rn in enumerate(s.resname) if rn == "ALA")
    mask = alanine_exclude_mask(s, [ala_resnum])
    assert not mask.any()  # Ala has nothing beyond CB to strip


def test_alanine_mutation_removes_sidechain_contacts_but_preserves_blocking():
    s = load_structure(CI2)
    ss = assign_secondary_structure(s)

    # Pick a residue with a real side chain (not already Ala/Gly/Pro).
    resnum = next(int(s.author_resnum[i]) for i, rn in enumerate(s.resname) if rn not in ("ALA", "GLY", "PRO"))

    cm_wt = compute_contact_map(s)
    bm_wt = build_blocks(ss, cm_wt, block_size=4)

    exclude = alanine_exclude_mask(s, [resnum])
    assert exclude.sum() > 0  # some side-chain atoms were actually excluded

    cm_mut = compute_contact_map(s, exclude_atoms=exclude)
    bm_mut = build_blocks(ss, cm_mut, block_size=4)

    # Mutation must never change secondary structure -> never change blocking.
    assert bm_mut.nblocks == bm_wt.nblocks
    assert np.array_equal(bm_mut.block_of_residue, bm_wt.block_of_residue)

    # Total contact count should drop (or stay equal in a pathological case
    # with zero contacts for that side chain), never increase.
    assert cm_mut.srcont.sum() <= cm_wt.srcont.sum()


def test_run_alanine_scan_smoke():
    s = load_structure(CI2, ph=7.0)
    ss = assign_secondary_structure(s)
    params = WSMEParams.soluble_protein_defaults()
    positions = scannable_positions(s)[:4]

    result = run_alanine_scan(s, ss, params, positions, block_size=4)
    nb = result.wt_chi_plus.shape[0]
    assert result.wt_chi_plus.shape == (nb, nb)
    assert set(result.mean_ddg_vector.keys()) == set(positions)
    for v in result.mean_ddg_vector.values():
        assert v.shape == (nb,)
    assert result.MR_mean.shape == (nb,)
    assert result.MR_std.shape == (nb,)
    # no +-inf anywhere (the near-zero-probability masking in compute_coupling
    # must apply to chi_plus itself, not just the symmetrized coupling matrix)
    for v in result.mean_ddg_vector.values():
        assert not np.any(np.isinf(v))


def test_run_alanine_scan_defaults_to_every_scannable_position():
    s = load_structure(CI2, ph=7.0)
    ss = assign_secondary_structure(s)
    params = WSMEParams.soluble_protein_defaults()

    # Cap heavily via max_positions so the "scan everything" default path
    # is exercised without paying for a full CI2 scan in the test suite.
    result = run_alanine_scan(s, ss, params, positions=None, max_positions=3, block_size=4)
    assert len(result.positions) == 3
    assert set(result.positions) <= set(scannable_positions(s))


def test_subsample_positions_is_evenly_spaced_and_covers_endpoints():
    positions = list(range(100, 200))  # 100 positions
    sub = subsample_positions(positions, 5)
    assert len(sub) == 5
    assert sub[0] == positions[0]
    assert sub[-1] == positions[-1]
    assert sub == sorted(sub)  # stays in sequence order


def test_subsample_positions_noop_when_cap_exceeds_length():
    positions = [1, 2, 3]
    assert subsample_positions(positions, 10) == positions


def test_estimate_scan_seconds_scales_linearly():
    assert estimate_scan_seconds(0, seconds_per_position=8.0) == 8.0  # just the WT baseline
    assert estimate_scan_seconds(10, seconds_per_position=8.0) == 88.0


def test_top_hits_and_distance_profile():
    s = load_structure(CI2, ph=7.0)
    ss = assign_secondary_structure(s)
    params = WSMEParams.soluble_protein_defaults()
    positions = scannable_positions(s)[:5]

    result = run_alanine_scan(s, ss, params, positions, block_size=4)

    hits = result.top_hits(n=3)
    assert len(hits) == 3
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)  # ranked descending

    resnum = positions[0]
    dist, ddg = result.ddg_vs_distance(resnum)
    nb = result.wt_chi_plus.shape[0]
    assert dist.shape == (nb,)
    assert ddg.shape == (nb,)
    # distance from a block to itself must be 0 (its own centroid)
    own_block = result.block_of_position[resnum]
    assert dist[own_block] == 0.0


def test_ph_sensitivity_table_ranks_by_swing_across_ph():
    params = WSMEParams.soluble_protein_defaults()
    scan_by_ph = {}
    for ph in (7.0, 5.0):
        s = load_structure(CI2, ph=ph)
        ss = assign_secondary_structure(s)
        positions = scannable_positions(s)[:5]
        scan_by_ph[ph] = run_alanine_scan(s, ss, params, positions, block_size=4)

    rows = ph_sensitivity_table(scan_by_ph, n=3)
    assert len(rows) > 0
    for row in rows:
        assert set(row["scores_by_ph"].keys()) == {7.0, 5.0}
        assert row["ph_spread"] >= 0.0
    # descending by ph_spread
    spreads = [r["ph_spread"] for r in rows]
    assert spreads == sorted(spreads, reverse=True)
    # every reported resnum must have appeared in at least one pH's top hits
    expected = set()
    for scan in scan_by_ph.values():
        expected.update(r for r, _ in scan.top_hits(3))
    assert set(r["resnum"] for r in rows) == expected


def _multi_ph_scan(n_positions=6, ph_values=(7.0, 5.0, 3.5)):
    params = WSMEParams.soluble_protein_defaults()
    scan_by_ph = {}
    for ph in ph_values:
        s = load_structure(CI2, ph=ph)
        ss = assign_secondary_structure(s)
        positions = scannable_positions(s)[:n_positions]
        scan_by_ph[ph] = run_alanine_scan(s, ss, params, positions, block_size=4)
    return scan_by_ph


def test_residue_ph_features_shape_and_nan_handling():
    scan_by_ph = _multi_ph_scan(n_positions=6, ph_values=(7.0, 5.0, 3.5))
    resnums, features, magnitude, ph_spread = residue_ph_features(scan_by_ph)

    n_residues = 6
    nblocks = next(iter(scan_by_ph.values())).wt_chi_plus.shape[0]
    assert resnums.shape == (n_residues,)
    assert features.shape == (n_residues, nblocks * 3)
    assert magnitude.shape == (n_residues,)
    assert ph_spread.shape == (n_residues,)
    assert not np.any(np.isnan(features))  # NaN coupling entries must be zeroed, not propagated
    assert np.all(magnitude >= 0.0)
    assert np.all(ph_spread >= 0.0)


def test_pca_cluster_residues_shapes_and_determinism():
    scan_by_ph = _multi_ph_scan(n_positions=8, ph_values=(7.0, 5.0, 3.5))
    _, features, _, _ = residue_ph_features(scan_by_ph)

    coords, labels, evr = pca_cluster_residues(features, n_components=2, n_clusters=3, seed=0)
    assert coords.shape == (8, 2)
    assert labels.shape == (8,)
    assert len(evr) == 2
    assert set(labels) <= set(range(3))

    # Same seed -> identical clustering (deterministic, reproducible for a report)
    coords2, labels2, _ = pca_cluster_residues(features, n_components=2, n_clusters=3, seed=0)
    assert np.allclose(coords, coords2)
    assert np.array_equal(labels, labels2)


def test_pca_cluster_residues_caps_k_to_n_residues():
    scan_by_ph = _multi_ph_scan(n_positions=3, ph_values=(7.0, 5.0))
    _, features, _, _ = residue_ph_features(scan_by_ph)
    # more clusters requested than residues available -- must not error
    coords, labels, _ = pca_cluster_residues(features, n_clusters=10, seed=0)
    assert coords.shape[0] == 3
    assert len(set(labels)) <= 3


def test_ph_cluster_table_rows_match_features_and_are_sorted_by_magnitude():
    scan_by_ph = _multi_ph_scan(n_positions=6, ph_values=(7.0, 5.0, 3.5))
    rows = ph_cluster_table(scan_by_ph, n_clusters=3, seed=0)

    resnums, _, magnitude, ph_spread = residue_ph_features(scan_by_ph)
    assert set(r["resnum"] for r in rows) == set(int(r) for r in resnums)

    magnitudes = [r["magnitude"] for r in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)

    by_resnum = {r["resnum"]: r for r in rows}
    for resnum, expected_mag, expected_spread in zip(resnums, magnitude, ph_spread):
        row = by_resnum[int(resnum)]
        assert row["magnitude"] == expected_mag
        assert row["ph_spread"] == expected_spread
