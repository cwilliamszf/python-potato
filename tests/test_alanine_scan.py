import numpy as np
from pathlib import Path

from wsme_gpcr.alanine_scan import (
    alanine_exclude_mask,
    estimate_scan_seconds,
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
