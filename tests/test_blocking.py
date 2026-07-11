import numpy as np

from wsme_gpcr.blocking import build_blocks, _run_length_chunks
from wsme_gpcr.contacts import ContactMap


def _empty_contact_map(nres):
    return ContactMap(nres=nres, srcont=np.zeros((nres, nres), dtype=np.int64), elec_pairs=np.zeros((0, 5)))


def test_run_length_chunks():
    mask = np.array([True, True, False, False, False, True])
    chunks = list(_run_length_chunks(mask))
    assert chunks == [(0, 2), (2, 5), (5, 6)]


def test_full_blocks_exact_multiple():
    # 8 structured residues, block_size 4 -> exactly two blocks, no residual.
    mask = np.array([True] * 8)
    bm = build_blocks(mask, _empty_contact_map(8), block_size=4)
    assert bm.nblocks == 2
    assert list(bm.block_of_residue) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_singleton_residual_merges_into_previous_block_after_first_full_block():
    # helix run of 5 (block_size 4): one full block + 1 residual residue.
    # Since a full block was already formed in this run, the residual
    # residue should merge into that same block (not start a new one).
    mask = np.array([True] * 5)
    bm = build_blocks(mask, _empty_contact_map(5), block_size=4)
    assert bm.nblocks == 1
    assert list(bm.block_of_residue) == [0, 0, 0, 0, 0]


def test_singleton_residual_before_any_full_block_gets_its_own_block():
    # A lone residue chunk of length 1 occurring before any full block has
    # ever been formed anywhere in the protein gets its own block id.
    mask = np.array([True, False, False, False, False])  # chunk lengths: 1, 4
    bm = build_blocks(mask, _empty_contact_map(5), block_size=4)
    # chunk0 (len1, no full block yet) -> new block 0
    # chunk1 (len4, exactly one full block) -> block 1
    assert bm.nblocks == 2
    assert list(bm.block_of_residue) == [0, 1, 1, 1, 1]


def test_residual_greater_than_one_gets_its_own_block():
    mask = np.array([True] * 6)  # block_size 4 -> one full block + residual 2
    bm = build_blocks(mask, _empty_contact_map(6), block_size=4)
    assert bm.nblocks == 2
    assert list(bm.block_of_residue) == [0, 0, 0, 0, 1, 1]


def test_block_cmap_aggregation():
    nres = 6
    cm = _empty_contact_map(nres)
    cm.srcont[0, 5] = 3  # residue 0 (block 0) <-> residue 5 (block 1)
    cm.srcont[1, 2] = 2  # both in block 0
    mask = np.array([True] * 6)
    bm = build_blocks(mask, cm, block_size=4)
    assert bm.nblocks == 2
    assert bm.block_cmap[0, 0] == 2  # intra-block-0 contact
    assert bm.block_cmap[0, 1] == 3  # cross block 0/1 contact
