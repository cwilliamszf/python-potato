"""Residue-to-block partitioning and block-level contact aggregation.

Ports the blocking logic at the end of ``cmapCalcElecBlock.m``. Residues
are first split into maximal runs of structured (H/E/G) vs. unstructured
(coil) residues, then each run is chopped into chunks of ``block_size``
residues (with a remainder chunk absorbing what's left over). This is a
literal port, including one MATLAB quirk preserved on purpose: a
leftover chunk of exactly 1 residue is merged into the *previous* block
rather than becoming its own block, for every run in the protein except
possibly the very first residue of the chain (before any full block has
ever been formed). This avoids spurious singleton blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contacts import ContactMap


@dataclass
class BlockModel:
    nres: int
    nblocks: int
    block_size: int
    block_of_residue: np.ndarray  # (nres,) 0-based block index per residue
    block_residue_range: np.ndarray  # (nblocks, 2) [first_res, last_res] per block
    block_cmap: np.ndarray  # (nblocks, nblocks) upper-triangular VdW contact counts
    block_elec: np.ndarray  # (K, 5): [blockA, blockB, dist, seqsep, energy_vacuum]


def _run_length_chunks(mask: np.ndarray):
    """Maximal runs of a constant boolean value. Yields (start, end_exclusive)."""
    n = len(mask)
    i = 0
    while i < n:
        j = i
        while j < n and mask[j] == mask[i]:
            j += 1
        yield i, j
        i = j


def build_blocks(structured_mask: np.ndarray, contact_map: ContactMap, block_size: int = 4) -> BlockModel:
    nres = contact_map.nres
    if len(structured_mask) != nres:
        raise ValueError("structured_mask length must equal nres")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    block_of_residue = np.full(nres, -1, dtype=int)
    block_id = -1
    any_full_block_formed = False
    pos = 0
    for cs, ce in _run_length_chunks(structured_mask):
        length = ce - cs
        n_full = length // block_size
        residual = length % block_size
        for _ in range(n_full):
            block_id += 1
            block_of_residue[pos:pos + block_size] = block_id
            pos += block_size
            any_full_block_formed = True
        if residual == 1:
            if not any_full_block_formed:
                block_id += 1
            block_of_residue[pos] = block_id
            pos += 1
        elif residual > 1:
            block_id += 1
            block_of_residue[pos:pos + residual] = block_id
            pos += residual

    if pos != nres or np.any(block_of_residue < 0):
        raise RuntimeError("internal error: blocking did not cover all residues")

    nblocks = block_id + 1

    block_residue_range = np.zeros((nblocks, 2), dtype=int)
    for b in range(nblocks):
        idx = np.where(block_of_residue == b)[0]
        block_residue_range[b] = [idx.min(), idx.max()]

    rows, cols = np.nonzero(contact_map.srcont)
    block_cmap = np.zeros((nblocks, nblocks), dtype=np.int64)
    if len(rows):
        vals = contact_map.srcont[rows, cols]
        brows = block_of_residue[rows]
        bcols = block_of_residue[cols]
        np.add.at(block_cmap, (brows, bcols), vals)

    if len(contact_map.elec_pairs):
        resi = contact_map.elec_pairs[:, 0].astype(int)
        resj = contact_map.elec_pairs[:, 1].astype(int)
        ba = block_of_residue[resi]
        bb = block_of_residue[resj]
        lo = np.minimum(ba, bb)
        hi = np.maximum(ba, bb)
        block_elec = np.column_stack([lo, hi, contact_map.elec_pairs[:, 2], contact_map.elec_pairs[:, 3], contact_map.elec_pairs[:, 4]])
    else:
        block_elec = np.zeros((0, 5))

    return BlockModel(
        nres=nres,
        nblocks=nblocks,
        block_size=block_size,
        block_of_residue=block_of_residue,
        block_residue_range=block_residue_range,
        block_cmap=block_cmap,
        block_elec=block_elec,
    )
