"""Residue(block)-residue(block) coupling free-energy analysis.

Ports the "coupling calculation" section of ``FesCalc_Block_full.m``
(the ``chipluswt`` / ``chiminuswt`` / ``dGwtcp`` matrices, i.e. the
``CouplingMat`` plotted in ``Plot_Imp_Variables.m``). This measures how
thermodynamically coupled the folding of block j is to the folding of
block k: positive values mean j and k tend to fold together (cooperative/
allosterically coupled), near zero means independent, negative means
anti-correlated (folding one tends to unfold the other).

The MATLAB code frames this as a per-residue-perturbation calculation
(``pert``/``permag``), but as shipped ``pert = nres+1`` is a no-op
sentinel (``npert == 1``), so the only quantity the reference code
actually ever produces is this single unperturbed (wild-type) coupling
matrix -- computed from co-occurrence statistics within the equilibrium
ensemble, no perturbation needed. That's what's implemented here.

For a single microstate (a folded SSA segment, or a folded DSA/DSAw/L
segment pair), every block in the folded set is folded and every other
block is unfolded -- deterministically, within that microstate. So its
contribution to the joint distribution of (block j folded?, block k
folded?) is exactly its equilibrium weight, added to whichever one of
the four (folded/folded, folded/unfolded, unfolded/folded,
unfolded/unfolded) quadrants (j, k) falls into. Both the folded set and
its complement are unions of a small, fixed number of contiguous block
ranges (at most 2 for an SSA segment, at most 3 for a DSA/DSAw/L pair),
so each quadrant can be built directly from a handful of rectangle
region-pairs -- no per-microstate O(block_count^2) work, and critically,
*no subtraction of large aggregate sums*. An earlier version of this
code derived the folded/unfolded and unfolded/unfolded quadrants as
``P(j folded) - P(j,k both folded)`` etc.; that's algebraically correct
but catastrophically cancels whenever a block is folded in nearly every
populated microstate (pF close to 0 or 1, which is common for core
residues) -- exactly the failure mode the MATLAB reference avoids by
summing state probabilities directly rather than subtracting aggregates.
Direct rectangle accumulation of every quadrant sidesteps that the same
way: every quantity is a sum of non-negative terms, so ordinary
floating-point summation error applies, not cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blocking import BlockModel
from .structure import Structure
from .wsme import WSMEParams, _build_topology, _evaluate, compute_block_zvec


@dataclass
class CouplingResult:
    p_folded: np.ndarray  # (nblocks,) marginal P(block folded)
    p_folded_folded: np.ndarray  # (nblocks, nblocks) joint P(j folded, k folded)
    p_folded_unfolded: np.ndarray  # (nblocks, nblocks) joint P(j folded, k unfolded)
    p_unfolded_unfolded: np.ndarray  # (nblocks, nblocks) joint P(j unfolded, k unfolded)
    chi_plus: np.ndarray  # (nblocks, nblocks) kJ/mol; -dG(j folds | k folded)
    chi_minus: np.ndarray  # (nblocks, nblocks) kJ/mol; -dG(j folds | k unfolded)
    coupling_free_energy: np.ndarray  # (nblocks, nblocks) kJ/mol; chi_plus - chi_minus
    zfin: float


def _add_rect(diff: np.ndarray, r0, r1, c0, c1, w):
    """2D range-update: add w to diff's implied matrix over rows [r0,r1]
    and cols [c0,c1] (inclusive) for every entry (vectorized), via 4
    corner updates. A single 2D cumsum over `diff` afterwards recovers
    the accumulated matrix. r0<=r1 and c0<=c1 must hold for every entry;
    callers must pre-filter out empty (invalid) ranges."""
    np.add.at(diff, (r0, c0), w)
    np.add.at(diff, (r0, c1 + 1), -w)
    np.add.at(diff, (r1 + 1, c0), -w)
    np.add.at(diff, (r1 + 1, c1 + 1), w)


def _accumulate(diff, from_regions, to_regions, w):
    """Add w to diff for every (j, k) with j in any 'from' region and k
    in any 'to' region, for every state (vectorized across states).
    Each region is (start_arr, end_arr, valid_mask)."""
    for fs, fe, fvalid in from_regions:
        for ts, te, tvalid in to_regions:
            valid = fvalid & tvalid
            if np.any(valid):
                _add_rect(diff, fs[valid], fe[valid], ts[valid], te[valid], w[valid])


def compute_coupling(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray, params: WSMEParams = None) -> CouplingResult:
    if params is None:
        params = WSMEParams()

    nb = block_model.nblocks
    zvec = compute_block_zvec(structure, block_model, ss_mask, params)
    zvalc = np.exp((params.DS - params.DDS) / params.R)
    topo = _build_topology(block_model)

    fes_num, _, _, _, _, raw = _evaluate(topo, block_model, zvec, zvalc, params, need_landscape=False, need_raw=True)
    zfin = float(fes_num.sum())
    if zfin <= 0 or not np.isfinite(zfin):
        raise RuntimeError("Partition function is zero/invalid; check contacts/parameters")

    diff_ff = np.zeros((nb + 1, nb + 1))
    diff_fu = np.zeros((nb + 1, nb + 1))
    diff_uu = np.zeros((nb + 1, nb + 1))

    last = nb - 1
    true_mask = lambda n: np.ones(n, dtype=bool)  # noqa: E731

    # ---- SSA: one folded segment [a, b]; complement is up to 2 intervals ----
    seg_a, seg_b, w1 = raw["seg_a"], raw["seg_b"], raw["w1"]
    n = len(seg_a)
    A = (seg_a, seg_b, true_mask(n))
    L = (np.zeros(n, dtype=int), seg_a - 1, seg_a >= 1)
    R = (seg_b + 1, np.full(n, last, dtype=int), seg_b <= last - 1)

    _accumulate(diff_ff, [A], [A], w1)
    _accumulate(diff_fu, [A], [L, R], w1)
    _accumulate(diff_uu, [L, R], [L, R], w1)

    # ---- DSA / DSAw/L: two folded segments A, B; complement is up to 3 intervals ----
    a1, b1, a2, b2, w_pair = raw["pair_a1"], raw["pair_b1"], raw["pair_a2"], raw["pair_b2"], raw["w_pair"]
    m = len(a1)
    if m:
        A = (a1, b1, true_mask(m))
        B = (a2, b2, true_mask(m))
        I0 = (np.zeros(m, dtype=int), a1 - 1, a1 >= 1)
        I1 = (b1 + 1, a2 - 1, true_mask(m))  # always non-empty: gap >= 1 block by construction
        I2 = (b2 + 1, np.full(m, last, dtype=int), b2 <= last - 1)

        _accumulate(diff_ff, [A, B], [A, B], w_pair)
        _accumulate(diff_fu, [A, B], [I0, I1, I2], w_pair)
        _accumulate(diff_uu, [I0, I1, I2], [I0, I1, I2], w_pair)

    FF = np.cumsum(np.cumsum(diff_ff, axis=0), axis=1)[:nb, :nb] / zfin
    FU = np.cumsum(np.cumsum(diff_fu, axis=0), axis=1)[:nb, :nb] / zfin
    UU = np.cumsum(np.cumsum(diff_uu, axis=0), axis=1)[:nb, :nb] / zfin
    # Genuinely-zero joint probabilities (e.g. a block folded in every
    # populated microstate makes "that block unfolded" a true zero-probability
    # event) can come out of the two nested cumsums as a tiny negative float
    # rather than exactly 0.0 -- machine-epsilon-scale rounding accumulated
    # over many +/-w corner updates, not a real negative probability. Clipped
    # to exactly 0 so a later 0/0 reads as "undefined" (NaN) consistently on
    # both sides of a pair, instead of "noise / noise" producing an arbitrary
    # finite-looking value or an inf on one side and NaN on the other.
    FF = np.clip(FF, 0.0, 1.0)
    FU = np.clip(FU, 0.0, 1.0)
    UU = np.clip(UU, 0.0, 1.0)
    UF = FU.T  # P(j unfolded, k folded) = P(k folded, j unfolded) relabeled

    pF = np.diag(FF).copy()

    R_, T = params.R, params.T
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_plus = R_ * T * np.log(FF / UF)
        chi_minus = R_ * T * np.log(FU / UU)
        # Computed as one combined log-ratio rather than chi_plus - chi_minus:
        # the argument (UU*FF)/(FU*UF) is bit-for-bit identical under j<->k
        # (numerator trivially symmetric; denominator is the same product of
        # the same two numbers either way), so this stays exactly symmetric
        # even at exact zeros where chi_plus/chi_minus individually hit
        # +-inf and a naive difference could collapse to inf on one side and
        # nan (inf - inf) on the other.
        coupling_free_energy = R_ * T * np.log((UU * FF) / (FU * UF))

    # A joint quadrant near float64's noise floor relative to the largest
    # accumulated weight (individual microstate weights can span up to ~60
    # orders of magnitude before normalization by zfin) can't be resolved
    # reliably even by the direct-accumulation approach above: the 2D
    # difference-array corner updates that recover it still sum large
    # +w/-w corrections whose net is a tiny value, so precision below
    # roughly 1e-6 in any of the four quadrants isn't trustworthy. Rather
    # than report a coupling free energy that's actually numerical noise
    # (and can differ between (j,k) and (k,j) by O(1) kJ/mol at that
    # scale), mark it undefined. min_quad is symmetric under j<->k by
    # construction (it's the elementwise min over the same four numbers,
    # {FF[j,k], FU[j,k], FU[k,j], UU[j,k]}, for both (j,k) and (k,j)), so
    # this masks symmetrically.
    _NEAR_ZERO_PROB = 1e-6
    min_quad = np.minimum(np.minimum(FF, FU), np.minimum(UU, UF))
    coupling_free_energy = np.where(min_quad < _NEAR_ZERO_PROB, np.nan, coupling_free_energy)
    # chi_plus = RT*ln(FF/UF) only involves FF and UF; chi_minus only FU and
    # UU -- mask each against just the quadrants it actually uses, rather
    # than the all-four min, so a chi_plus entry isn't discarded over a
    # near-zero UU it doesn't even depend on.
    chi_plus = np.where(np.minimum(FF, UF) < _NEAR_ZERO_PROB, np.nan, chi_plus)
    chi_minus = np.where(np.minimum(FU, UU) < _NEAR_ZERO_PROB, np.nan, chi_minus)

    np.fill_diagonal(chi_plus, np.nan)
    np.fill_diagonal(chi_minus, np.nan)
    np.fill_diagonal(coupling_free_energy, np.nan)

    return CouplingResult(
        p_folded=pF,
        p_folded_folded=FF,
        p_folded_unfolded=FU,
        p_unfolded_unfolded=UU,
        chi_plus=chi_plus,
        chi_minus=chi_minus,
        coupling_free_energy=coupling_free_energy,
        zfin=zfin,
    )
