from pathlib import Path

import numpy as np

from wsme_gpcr.pipeline import DEFAULT_PH_VALUES, run_pipeline, run_pipeline_multi_ph
from wsme_gpcr.wsme import WSMEParams

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"


def test_run_pipeline_smoke():
    pr = run_pipeline(CI2, ph=7.0, params=WSMEParams.soluble_protein_defaults())
    assert pr.structure.nres == 65
    assert pr.block_model.nblocks > 0
    assert pr.result.zfin > 0
    assert pr.dsc_result is None
    assert pr.coupling_result is None


def test_run_pipeline_with_coupling():
    pr = run_pipeline(CI2, ph=7.0, params=WSMEParams.soluble_protein_defaults(), with_coupling=True)
    nb = pr.block_model.nblocks
    c = pr.coupling_result
    assert c.coupling_free_energy.shape == (nb, nb)
    assert c.zfin == pr.result.zfin
    # symmetric off-diagonal (NaN where a joint quadrant has zero population,
    # e.g. a block folded in every populated microstate -- the coupling free
    # energy is genuinely undefined there, not just hard to compute), and
    # self-coupling undefined (NaN) on the diagonal
    off_diag = ~np.eye(nb, dtype=bool)
    assert np.allclose(c.coupling_free_energy[off_diag], c.coupling_free_energy.T[off_diag], equal_nan=True)
    assert np.all(np.isnan(np.diag(c.coupling_free_energy)))
    # marginal P(folded) must lie in [0, 1]
    assert np.all((c.p_folded >= 0) & (c.p_folded <= 1))


def test_run_pipeline_multi_ph_covers_all_default_values():
    results = run_pipeline_multi_ph(CI2, params=WSMEParams.soluble_protein_defaults())
    assert set(results.keys()) == set(DEFAULT_PH_VALUES)
    for ph, pr in results.items():
        assert pr.ph == ph
        assert pr.result.zfin > 0

    # Different pH values should generally give different partition functions
    # (charge assignment changes both the contact map and electrostatics).
    zfins = {ph: pr.result.zfin for ph, pr in results.items()}
    assert len(set(zfins.values())) > 1
