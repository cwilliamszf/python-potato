"""Validates compute_coupling's 2D-difference-array joint-probability
accumulation against a literal brute-force enumeration of every
SSA/DSA/DSAw-L microstate's folded-block set, on small random systems."""

import numpy as np
import pytest

from wsme_gpcr.blocking import BlockModel
from wsme_gpcr.coupling import compute_coupling
from wsme_gpcr.wsme import WSMEParams, _build_topology, _debye_screened_emap, _evaluate


def _make_block_model(nb, cmap, emap, rng):
    nres = nb
    block_of_residue = np.arange(nb)
    block_residue_range = np.column_stack([block_of_residue, block_of_residue])

    ii, jj = np.triu_indices(nb, k=1)
    evals = emap[ii, jj]
    nz = evals != 0
    dist = rng.uniform(3.0, 10.0, size=nz.sum())
    seqsep = np.abs(ii[nz] - jj[nz])
    block_elec = np.column_stack([ii[nz], jj[nz], dist, seqsep, evals[nz]])

    return BlockModel(
        nres=nres,
        nblocks=nb,
        block_size=1,
        block_of_residue=block_of_residue,
        block_residue_range=block_residue_range,
        block_cmap=cmap,
        block_elec=block_elec if len(block_elec) else np.zeros((0, 5)),
    )


def brute_force_joint_probs(cmap, emap, zvec, zvalc, ene, DCp, T, Tref, R):
    nb = len(zvec)
    dcp_term = ene + DCp * (T - Tref) - T * DCp * np.log(T / Tref)
    ff_num = np.zeros((nb, nb))
    fu_num = np.zeros((nb, nb))
    uu_num = np.zeros((nb, nb))
    zfin = 0.0

    def add_state(blocks, w):
        nonlocal ff_num, fu_num, uu_num
        folded = np.zeros(nb, dtype=bool)
        folded[sorted(blocks)] = True
        unfolded = ~folded
        idx_f = np.where(folded)[0]
        idx_u = np.where(unfolded)[0]
        ff_num[np.ix_(idx_f, idx_f)] += w
        fu_num[np.ix_(idx_f, idx_u)] += w
        uu_num[np.ix_(idx_u, idx_u)] += w

    # SSA
    for i in range(nb):
        for length in range(1, nb - i + 1):
            j = i + length - 1
            ncont = cmap[i:j + 1, i:j + 1].sum()
            eneE = emap[i:j + 1, i:j + 1].sum()
            stabE = ncont * dcp_term
            zprod = np.prod(zvec[i:j + 1])
            w = np.exp(-(stabE + eneE) / (R * T)) * zprod
            zfin += w
            add_state(range(i, j + 1), w)

    # DSA + DSAw/L
    for i in range(nb):
        for iin in range(1, nb - i + 1):
            for j in range(i + iin + 1, nb):
                for jin in range(1, nb - j + 1):
                    s1, e1 = i, i + iin - 1
                    s2, e2 = j, j + jin - 1
                    ncont1 = cmap[s1:e1 + 1, s1:e1 + 1].sum()
                    ncont2 = cmap[s2:e2 + 1, s2:e2 + 1].sum()
                    eneE1 = emap[s1:e1 + 1, s1:e1 + 1].sum()
                    eneE2 = emap[s2:e2 + 1, s2:e2 + 1].sum()
                    stabE = (ncont1 + ncont2) * dcp_term
                    eneE = eneE1 + eneE2
                    zprod = np.prod(zvec[s1:e1 + 1]) * np.prod(zvec[s2:e2 + 1])
                    w_dsa = np.exp(-(stabE + eneE) / (R * T)) * zprod

                    vv = list(range(s1, e1 + 1)) + list(range(s2, e2 + 1))
                    ncont_all = cmap[np.ix_(vv, vv)].sum()
                    eneE_all = emap[np.ix_(vv, vv)].sum()
                    stabE_all = ncont_all * dcp_term
                    cross_cmap = cmap[s1:e1 + 1, s2:e2 + 1].sum()
                    cross_emap = emap[s1:e1 + 1, s2:e2 + 1].sum()
                    if cross_cmap != 0 or cross_emap != 0:
                        gap = j - (i + iin)
                        w_dsawl = np.exp(-(stabE_all + eneE_all) / (R * T)) * zprod * (zvalc ** gap)
                    else:
                        w_dsawl = 0.0

                    w_pair = w_dsa + w_dsawl
                    zfin += w_pair
                    add_state(vv, w_pair)

    return ff_num, fu_num, uu_num, zfin


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_coupling_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    nb = 6

    cmap = rng.integers(0, 4, size=(nb, nb)).astype(float)
    cmap = np.triu(cmap, k=1)

    params = WSMEParams(T=305.0, ene=-0.05, DS=-0.012, DCp=-0.0004, IS=0.1, dielectric=4.0)
    zval = np.exp(params.DS / params.R)
    zvalc = np.exp((params.DS - params.DDS) / params.R)
    zvec = rng.choice([zval, zvalc], size=nb)

    emap_raw = np.triu(rng.choice([0, 0, 0, 1.5, -1.2], size=(nb, nb)), k=1)
    bm = _make_block_model(nb, cmap, emap_raw, rng)

    emap_screened = _debye_screened_emap(bm, params.T, params.IS, params.dielectric)
    ff_num_bf, fu_num_bf, uu_num_bf, zfin_bf = brute_force_joint_probs(
        cmap, emap_screened, zvec, zvalc, params.ene, params.DCp, params.T, params.Tref, params.R
    )
    ff_bf, fu_bf, uu_bf = ff_num_bf / zfin_bf, fu_num_bf / zfin_bf, uu_num_bf / zfin_bf

    # compute_coupling derives zvec internally from structure/ss_mask; to test
    # against an arbitrary zvec (as used by the brute-force reference above),
    # replicate its call path directly (same helpers compute_coupling uses)
    # rather than going through the public structure-based API.
    from wsme_gpcr.coupling import _accumulate

    topo = _build_topology(bm)
    fes_num, _, _, _, _, raw = _evaluate(topo, bm, zvec, zvalc, params, need_landscape=False, need_raw=True)
    zfin = float(fes_num.sum())
    assert zfin == pytest.approx(zfin_bf, rel=1e-9)

    diff_ff = np.zeros((nb + 1, nb + 1))
    diff_fu = np.zeros((nb + 1, nb + 1))
    diff_uu = np.zeros((nb + 1, nb + 1))
    last = nb - 1
    true_mask = lambda n: np.ones(n, dtype=bool)  # noqa: E731

    seg_a, seg_b, w1 = raw["seg_a"], raw["seg_b"], raw["w1"]
    n = len(seg_a)
    A = (seg_a, seg_b, true_mask(n))
    L = (np.zeros(n, dtype=int), seg_a - 1, seg_a >= 1)
    Rr = (seg_b + 1, np.full(n, last, dtype=int), seg_b <= last - 1)
    _accumulate(diff_ff, [A], [A], w1)
    _accumulate(diff_fu, [A], [L, Rr], w1)
    _accumulate(diff_uu, [L, Rr], [L, Rr], w1)

    a1, b1, a2, b2, w_pair = raw["pair_a1"], raw["pair_b1"], raw["pair_a2"], raw["pair_b2"], raw["w_pair"]
    m = len(a1)
    if m:
        A2 = (a1, b1, true_mask(m))
        B2 = (a2, b2, true_mask(m))
        I0 = (np.zeros(m, dtype=int), a1 - 1, a1 >= 1)
        I1 = (b1 + 1, a2 - 1, true_mask(m))
        I2 = (b2 + 1, np.full(m, last, dtype=int), b2 <= last - 1)
        _accumulate(diff_ff, [A2, B2], [A2, B2], w_pair)
        _accumulate(diff_fu, [A2, B2], [I0, I1, I2], w_pair)
        _accumulate(diff_uu, [I0, I1, I2], [I0, I1, I2], w_pair)

    ff = np.cumsum(np.cumsum(diff_ff, axis=0), axis=1)[:nb, :nb] / zfin
    fu = np.cumsum(np.cumsum(diff_fu, axis=0), axis=1)[:nb, :nb] / zfin
    uu = np.cumsum(np.cumsum(diff_uu, axis=0), axis=1)[:nb, :nb] / zfin
    uf = fu.T

    assert np.allclose(ff, ff_bf, rtol=1e-9, atol=1e-12)
    assert np.allclose(fu, fu_bf, rtol=1e-9, atol=1e-12)
    assert np.allclose(uu, uu_bf, rtol=1e-9, atol=1e-12)
    assert np.allclose(ff, ff.T, rtol=1e-9, atol=1e-12)  # joint P(j,k) must be symmetric
    # the four quadrants must partition the full ensemble for every (j, k)
    assert np.allclose(ff + fu + uf + uu, 1.0, atol=1e-9)
