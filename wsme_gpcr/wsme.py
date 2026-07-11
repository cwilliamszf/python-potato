"""Core blocked-WSME (bWSME) statistical mechanical engine.

Ports the partition-function enumeration in ``FesCalc_Block.m`` /
``FesCalc_Block_full.m`` (SSA + DSA + DSAw/L microstates, 1D/2D
free-energy profiles, residue folding probability vs. reaction
coordinate). The residue-residue coupling free-energy analysis from
``FesCalc_Block_full.m`` is intentionally out of scope (see README).

Physics, verbatim from the MATLAB reference:
  - A microstate is a set of contiguous "blocks" that are folded; every
    other block is unfolded. SSA microstates have one folded segment,
    DSA/DSAw/L microstates have two, separated by >=1 unfolded block.
  - Each folded block b contributes a multiplicative entropic factor
    z_b = exp(DS/R) (ordered) or exp((DS-DDS)/R) (coil/Gly, an extra
    entropic penalty), or 1 for a block containing only proline
    (already conformationally restricted).
  - Folded blocks contribute a van der Waals stabilization energy
    proportional to their native-contact count, with a heat-capacity
    (DCp) correction, plus a Debye-Hueckel-screened electrostatic term.
  - When the two segments of a DSA/DSAw/L microstate directly contact
    each other (nonzero cross contact/electrostatic energy), an
    additional loop-closure microstate (DSAw/L) is available on top of
    the independent-segment one (DSA), carrying an extra entropic
    penalty zvalc**gap for closing the intervening loop.

This module vectorizes what the MATLAB code computes with 2-4 nested
loops (recomputing submatrix sums from scratch every iteration) using
2D prefix sums ("summed area tables") for O(1) segment/cross-segment
energy lookups, so realistic GPCR-sized proteins (nblocks ~ 60-100)
finish in seconds rather than being computationally intractable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .blocking import BlockModel
from .structure import Structure

_R = 0.008314  # kJ/(mol.K)


@dataclass
class WSMEParams:
    """Model parameters. Defaults are the GPCR/membrane-protein preset
    used in AthiNaganathan/GPCR-Landscapes (dielectric=4).  For soluble
    proteins (as in AthiNaganathan/WSMEmodel's CI2 example) use
    ``WSMEParams.soluble_protein_defaults()`` instead (dielectric=29).
    """

    T: float = 310.0  # temperature, K
    ene: float = -48.2e-3  # vdW interaction energy per native contact, kJ/mol
    DS: float = -10e-3  # entropic cost per structured residue, kJ/mol/K
    DCp: float = -0.3579e-3  # heat capacity change per native contact, kJ/mol/K
    DDS: float = 6.0606e-3  # excess entropic cost for coil/Gly residues, kJ/mol/K
    IS: float = 0.1  # ionic strength, M
    dielectric: float = 4.0  # medium dielectric constant (4: membrane, 29: soluble)
    Tref: float = 385.0  # reference temperature for DCp term, K
    R: float = _R

    @classmethod
    def soluble_protein_defaults(cls) -> "WSMEParams":
        return cls(ene=-98e-3, DS=-14.5e-3, DCp=-0.3579e-3, dielectric=29.0)


@dataclass
class WSMEResult:
    n_values: np.ndarray  # (nblocks,) number of structured blocks, 1..nblocks
    fes: np.ndarray  # (nblocks,) kJ/mol, 1D free-energy profile vs n_values
    hv: int  # N-/C-terminal split point (in blocks) used for the 2D landscape
    fes2D: np.ndarray  # (hv+1, nblocks-hv+1) kJ/mol, [n_Nterm, n_Cterm]
    fpath: np.ndarray  # (nblocks, nblocks) [n_values index, block index] -> P(folded)
    zfin: float
    stats: dict = field(default_factory=dict)
    block_model: BlockModel = None


def _prefix_sum(mat: np.ndarray) -> np.ndarray:
    p = np.zeros((mat.shape[0] + 1, mat.shape[1] + 1), dtype=float)
    p[1:, 1:] = np.cumsum(np.cumsum(mat, axis=0), axis=1)
    return p


def _range_sum(prefix, r0, r1, c0, c1):
    """Inclusive-block-index sum of the original matrix over rows
    [r0..r1] and cols [c0..c1]. Args may be numpy arrays (elementwise)."""
    return prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1] - prefix[r1 + 1, c0] + prefix[r0, c0]


def _overlap(a, b, lo, hi):
    return np.maximum(0, np.minimum(b, hi) - np.maximum(a, lo) + 1)


def compute_block_zvec(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray, params: WSMEParams) -> np.ndarray:
    """Per-block entropic z-factor (product of per-residue z-factors).

    Per-residue rule (ports the disr/ppos logic in FesCalc_Block*.m):
      - proline residues: z = 1 (already conformationally restricted)
      - glycine residues, or any residue outside a helix/strand/3-10
        segment: z = exp((DS - DDS) / R)  (extra entropic penalty)
      - everything else: z = exp(DS / R)
    """
    zval = np.exp(params.DS / params.R)
    zvalc = np.exp((params.DS - params.DDS) / params.R)

    is_pro = np.array([r == "PRO" for r in structure.resname])
    is_gly = np.array([r == "GLY" for r in structure.resname])
    disordered = (is_gly | ~ss_mask) & ~is_pro

    zjj = np.where(is_pro, 1.0, np.where(disordered, zvalc, zval))

    zvec = np.ones(block_model.nblocks)
    for b in range(block_model.nblocks):
        residues = np.where(block_model.block_of_residue == b)[0]
        zvec[b] = np.prod(zjj[residues])
    return zvec


def _debye_screened_emap(block_model: BlockModel, T: float, IS: float, dielectric: float) -> np.ndarray:
    nb = block_model.nblocks
    emap = np.zeros((nb, nb), dtype=float)
    pairs = block_model.block_elec
    if len(pairs) == 0:
        return emap
    ISfac = 5.66 * np.sqrt(IS / T) * np.sqrt(80.0 / dielectric)
    a = pairs[:, 0].astype(int)
    b = pairs[:, 1].astype(int)
    dist = pairs[:, 2]
    energy = pairs[:, 4]
    screened = energy * np.exp(-ISfac * dist)
    np.add.at(emap, (a, b), screened)
    return emap


@dataclass
class _Topology:
    """Block-index combinatorics and cmap-derived sums, independent of
    temperature/IS/dielectric. Reused across a temperature sweep (DSC)
    without re-deriving the O(nblocks^4) pair enumeration each time."""

    nb: int
    seg_a: np.ndarray
    seg_b: np.ndarray
    seg_len: np.ndarray
    ncont_seg: np.ndarray
    iA: np.ndarray
    iB: np.ndarray
    cross_cmap: np.ndarray
    gap: np.ndarray
    cmap_prefix: np.ndarray


def _build_topology(block_model: BlockModel) -> _Topology:
    nb = block_model.nblocks
    cmap = block_model.block_cmap.astype(float)
    cmap_prefix = _prefix_sum(cmap)

    seg_a, seg_b = np.triu_indices(nb)
    seg_len = seg_b - seg_a + 1
    ncont_seg = _range_sum(cmap_prefix, seg_a, seg_b, seg_a, seg_b)

    mask = seg_a[None, :] >= seg_b[:, None] + 2  # rows=A, cols=B (B after A)
    iA, iB = np.where(mask)
    if len(iA):
        a1, b1 = seg_a[iA], seg_b[iA]
        a2, b2 = seg_a[iB], seg_b[iB]
        cross_cmap = _range_sum(cmap_prefix, a1, b1, a2, b2)
        gap = a2 - b1 - 1
    else:
        cross_cmap = np.zeros(0)
        gap = np.zeros(0, dtype=int)

    return _Topology(nb, seg_a, seg_b, seg_len, ncont_seg, iA, iB, cross_cmap, gap, cmap_prefix)


def _evaluate(topo: _Topology, block_model: BlockModel, zvec: np.ndarray, zvalc: float, params: WSMEParams,
              need_landscape: bool, need_raw: bool = False):
    """T-dependent energy evaluation given a fixed topology. Returns
    (fes_num, fes2D_num_or_None, diff_or_None, hv, stats, raw_or_None).

    ``raw``, when requested, exposes the per-segment SSA weights and
    per-pair DSA+DSAw/L weights (not yet binned by reaction coordinate)
    so a caller can do its own accumulation -- e.g. the coupling-matrix
    calculation in coupling.py needs the raw weights indexed by block
    range, not just the RC-binned totals."""
    nb = topo.nb
    T, R, Tref = params.T, params.R, params.Tref
    dcp_term = params.ene + params.DCp * (T - Tref) - T * params.DCp * np.log(T / Tref)

    logzvec = np.log(zvec)
    cumlogz = np.concatenate([[0.0], np.cumsum(logzvec)])

    emap = _debye_screened_emap(block_model, T, params.IS, params.dielectric)
    emap_prefix = _prefix_sum(emap)

    seg_a, seg_b, seg_len, ncont_seg = topo.seg_a, topo.seg_b, topo.seg_len, topo.ncont_seg
    eneE_seg = _range_sum(emap_prefix, seg_a, seg_b, seg_a, seg_b)
    stabE_seg = ncont_seg * dcp_term
    logzprod_seg = cumlogz[seg_b + 1] - cumlogz[seg_a]
    w1 = np.exp(-(stabE_seg + eneE_seg) / (R * T) + logzprod_seg)

    fes_num = np.zeros(nb + 1)
    np.add.at(fes_num, seg_len, w1)

    hv = int(round(nb / 2))
    fes2D_num = np.zeros((hv + 1, nb - hv + 1)) if need_landscape else None
    diff = np.zeros((nb + 1, nb + 1)) if need_landscape else None
    if need_landscape:
        nN_seg = _overlap(seg_a, seg_b, 0, hv - 1)
        nC_seg = _overlap(seg_a, seg_b, hv, nb - 1)
        np.add.at(fes2D_num, (nN_seg, nC_seg), w1)
        np.add.at(diff, (seg_len, seg_a), w1)
        np.add.at(diff, (seg_len, seg_b + 1), -w1)

    z_ssa = float(w1.sum())
    z_dsa = 0.0
    z_dsawl = 0.0
    n_states_dsawl = 0

    iA, iB = topo.iA, topo.iB
    if len(iA):
        a1, b1 = seg_a[iA], seg_b[iA]
        a2, b2 = seg_a[iB], seg_b[iB]
        w1_A, w1_B = w1[iA], w1[iB]
        base = w1_A * w1_B

        cross_emap = _range_sum(emap_prefix, a1, b1, a2, b2)
        interact = (topo.cross_cmap != 0) | (cross_emap != 0)

        correction = np.zeros_like(base)
        if np.any(interact):
            cc, ce, g = topo.cross_cmap[interact], cross_emap[interact], topo.gap[interact]
            correction[interact] = np.exp(-(cc * dcp_term + ce) / (R * T)) * (zvalc ** g)

        w_dsa = base
        w_dsawl = base * correction
        w_pair = w_dsa + w_dsawl
        pair_len = (b1 - a1 + 1) + (b2 - a2 + 1)

        np.add.at(fes_num, pair_len, w_pair)

        if need_landscape:
            nN_pair = _overlap(a1, b1, 0, hv - 1) + _overlap(a2, b2, 0, hv - 1)
            nC_pair = _overlap(a1, b1, hv, nb - 1) + _overlap(a2, b2, hv, nb - 1)
            np.add.at(fes2D_num, (nN_pair, nC_pair), w_pair)
            np.add.at(diff, (pair_len, a1), w_pair)
            np.add.at(diff, (pair_len, b1 + 1), -w_pair)
            np.add.at(diff, (pair_len, a2), w_pair)
            np.add.at(diff, (pair_len, b2 + 1), -w_pair)

        z_dsa = float(w_dsa.sum())
        z_dsawl = float(w_dsawl.sum())
        n_states_dsawl = int(np.count_nonzero(w_dsawl))

    zfin = float(fes_num.sum())
    stats = {
        "n_states_ssa": len(seg_a),
        "n_states_dsa": len(iA),
        "n_states_dsawl": n_states_dsawl,
        "pct_ssa": 100.0 * z_ssa / zfin if zfin else float("nan"),
        "pct_dsa": 100.0 * z_dsa / zfin if zfin else float("nan"),
        "pct_dsawl": 100.0 * z_dsawl / zfin if zfin else float("nan"),
    }

    raw = None
    if need_raw:
        if len(iA):
            w_pair = w_dsa + w_dsawl
        else:
            w_pair = np.zeros(0)
        raw = {
            "seg_a": seg_a, "seg_b": seg_b, "w1": w1,
            "pair_a1": seg_a[iA] if len(iA) else np.zeros(0, dtype=int),
            "pair_b1": seg_b[iA] if len(iA) else np.zeros(0, dtype=int),
            "pair_a2": seg_a[iB] if len(iA) else np.zeros(0, dtype=int),
            "pair_b2": seg_b[iB] if len(iA) else np.zeros(0, dtype=int),
            "w_pair": w_pair,
        }

    return fes_num, fes2D_num, diff, hv, stats, raw


def run_wsme(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray, params: WSMEParams = None) -> WSMEResult:
    if params is None:
        params = WSMEParams()

    nb = block_model.nblocks
    if nb > 120:
        warnings.warn(
            f"nblocks={nb} is large; DSA/DSAw/L enumeration scales roughly as "
            "nblocks^4 and may be slow/memory-heavy. Consider a larger block_size.",
            stacklevel=2,
        )

    zvec = compute_block_zvec(structure, block_model, ss_mask, params)
    zvalc = np.exp((params.DS - params.DDS) / params.R)

    topo = _build_topology(block_model)
    fes_num, fes2D_num, diff, hv, stats, _ = _evaluate(topo, block_model, zvec, zvalc, params, need_landscape=True)

    zfin = float(fes_num.sum())
    if zfin <= 0 or not np.isfinite(zfin):
        raise RuntimeError("Partition function is zero/invalid; check contacts/parameters")

    R, T = params.R, params.T
    with np.errstate(divide="ignore", invalid="ignore"):
        fes_full = -R * T * np.log(fes_num / zfin)
        fes2D_full = -R * T * np.log(fes2D_num / zfin)
        fpath_numerator = np.cumsum(diff, axis=1)[:, :nb]
        fpath_full = fpath_numerator / fes_num[:, None]

    return WSMEResult(
        n_values=np.arange(1, nb + 1),
        fes=fes_full[1:],
        hv=hv,
        fes2D=fes2D_full,
        fpath=fpath_full[1:, :],
        zfin=zfin,
        stats=stats,
        block_model=block_model,
    )


def partition_function(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray, params: WSMEParams,
                        topo: "_Topology" = None) -> float:
    """Total partition function Zfin at the given params, without the
    O(nblocks^2) 2D-landscape/Fpath bookkeeping. Used by dsc.py for fast
    temperature sweeps; pass a precomputed ``topo`` (from
    ``_build_topology``) to skip re-deriving the state combinatorics."""
    zvec = compute_block_zvec(structure, block_model, ss_mask, params)
    zvalc = np.exp((params.DS - params.DDS) / params.R)
    if topo is None:
        topo = _build_topology(block_model)
    fes_num, _, _, _, _, _ = _evaluate(topo, block_model, zvec, zvalc, params, need_landscape=False)
    return float(fes_num.sum())
