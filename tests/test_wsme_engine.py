"""Validates the vectorized WSME engine (wsme_gpcr.wsme) against a literal,
unvectorized brute-force enumeration of the same SSA/DSA/DSAw-L microstate
space (a direct translation of the nested loops in FesCalc_Block.m), on
small random synthetic systems where brute force is tractable.
"""

import numpy as np
import pytest

from wsme_gpcr.blocking import BlockModel
from wsme_gpcr.wsme import _build_topology, _evaluate, WSMEParams


def brute_force(cmap, emap, zvec, zvalc, ene, DCp, T, Tref, R):
    nb = len(zvec)
    dcp_term = ene + DCp * (T - Tref) - T * DCp * np.log(T / Tref)
    fes_num = np.zeros(nb + 1)
    hv = round(nb / 2)
    fes2D_num = np.zeros((hv + 1, nb - hv + 1))
    fpath_num = np.zeros((nb + 1, nb))

    def overlap(lo_seg, hi_seg, lo_r, hi_r):
        return max(0, min(hi_seg, hi_r) - max(lo_seg, lo_r) + 1)

    # SSA
    for i in range(nb):
        for length in range(1, nb - i + 1):
            j = i + length - 1
            ncont = cmap[i:j + 1, i:j + 1].sum()
            eneE = emap[i:j + 1, i:j + 1].sum()
            stabE = ncont * dcp_term
            zprod = np.prod(zvec[i:j + 1])
            w = np.exp(-(stabE + eneE) / (R * T)) * zprod
            fes_num[length] += w
            nN = overlap(i, j, 0, hv - 1)
            nC = overlap(i, j, hv, nb - 1)
            fes2D_num[nN, nC] += w
            fpath_num[length, i:j + 1] += w

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

                    n = iin + jin
                    w_pair = w_dsa + w_dsawl
                    fes_num[n] += w_pair
                    nN = overlap(s1, e1, 0, hv - 1) + overlap(s2, e2, 0, hv - 1)
                    nC = overlap(s1, e1, hv, nb - 1) + overlap(s2, e2, hv, nb - 1)
                    fes2D_num[nN, nC] += w_pair
                    fpath_num[n, s1:e1 + 1] += w_pair
                    fpath_num[n, s2:e2 + 1] += w_pair

    return fes_num, fes2D_num, fpath_num


def _make_block_model(nb, cmap, emap, rng):
    # Build a minimal BlockModel carrying only what _build_topology/_evaluate need.
    nres = nb  # 1 residue per block keeps this irrelevant to the math under test
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


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_vectorized_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    nb = 7

    cmap = rng.integers(0, 4, size=(nb, nb)).astype(float)
    cmap = np.triu(cmap, k=1)  # only off-diagonal upper triangle, as in real contact maps

    params = WSMEParams(T=305.0, ene=-0.05, DS=-0.012, DCp=-0.0004, IS=0.1, dielectric=4.0)
    zval = np.exp(params.DS / params.R)
    zvalc = np.exp((params.DS - params.DDS) / params.R)
    zvec = rng.choice([zval, zvalc], size=nb)

    # Sparse electrostatic energies (kJ/mol, pre-Debye-screening "vacuum" energy)
    emap_raw = np.triu(rng.choice([0, 0, 0, 1.5, -1.2], size=(nb, nb)), k=1)
    # Build a block model so we can reuse _debye_screened_emap exactly as the engine does.
    bm = _make_block_model(nb, cmap, emap_raw, rng)

    from wsme_gpcr.wsme import _debye_screened_emap
    emap_screened = _debye_screened_emap(bm, params.T, params.IS, params.dielectric)

    fes_bf, fes2D_bf, fpath_bf = brute_force(
        cmap, emap_screened, zvec, zvalc, params.ene, params.DCp, params.T, params.Tref, params.R
    )

    topo = _build_topology(bm)
    fes_num, fes2D_num, diff, hv, stats = _evaluate(topo, bm, zvec, zvalc, params, need_landscape=True)
    fpath_num = np.cumsum(diff, axis=1)[:, :nb]

    assert np.allclose(fes_num, fes_bf, rtol=1e-9, atol=1e-12)
    assert np.allclose(fes2D_num, fes2D_bf, rtol=1e-9, atol=1e-12)
    assert np.allclose(fpath_num, fpath_bf, rtol=1e-9, atol=1e-12)
    assert stats["n_states_ssa"] == nb * (nb + 1) // 2
