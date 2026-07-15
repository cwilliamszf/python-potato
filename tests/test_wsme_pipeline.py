from pathlib import Path

import numpy as np

from wsme_gpcr.pipeline import (
    DEFAULT_PH_VALUES,
    run_alanine_scan_pipeline,
    run_alanine_scan_pipeline_multi_ph,
    run_pipeline,
    run_pipeline_multi_ph,
)
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


def test_run_alanine_scan_pipeline_smoke():
    scan_pr = run_alanine_scan_pipeline(CI2, ph=7.0, params=WSMEParams.soluble_protein_defaults(), max_positions=3)
    assert scan_pr.ph == 7.0
    assert len(scan_pr.scan.positions) == 3
    nb = scan_pr.block_model.nblocks
    assert scan_pr.scan.wt_chi_plus.shape == (nb, nb)


def test_run_alanine_scan_pipeline_multi_ph_runs_independently_per_ph():
    ph_values = (7.0, 5.0)
    results = run_alanine_scan_pipeline_multi_ph(
        CI2, ph_values=ph_values, params=WSMEParams.soluble_protein_defaults(), max_positions=3,
    )
    assert set(results.keys()) == set(ph_values)
    for ph, scan_pr in results.items():
        assert scan_pr.ph == ph
        assert len(scan_pr.scan.positions) == 3

    # Different pH -> different structure/coupling, so the wild-type chi_plus
    # baselines should generally differ (not literally re-running the same scan).
    wt_matrices = [scan_pr.scan.wt_chi_plus for scan_pr in results.values()]
    assert not np.allclose(wt_matrices[0], wt_matrices[1], equal_nan=True)
