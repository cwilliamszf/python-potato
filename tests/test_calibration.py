from pathlib import Path

import numpy as np
import pytest

from wsme_gpcr.calibration import (
    CalibrationError,
    PAPER_TARGET_TM_K,
    PAPER_XI_MEAN_J_MOL,
    PAPER_XI_STD_J_MOL,
    XiTmCalibrationResult,
    calibrate_xi_isostability_mode,
    calibrate_xi_tm_mode,
    compute_delta_g_fold,
    find_cp_peaks_and_tm,
)
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.wsme import WSMEParams, WSMEResult

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"

# A bracket (J/mol) verified, by direct scan, to give a clean single Cp
# peak inside the default 280-360 K search grid at BOTH edges for CI2 with
# soluble_protein_defaults() -- NOT the paper's GPCR-specific -80/-20
# bracket (CI2 isn't a GPCR; this is a fast, real, non-mocked plumbing
# fixture only, not a scientific validation -- that happens on real GPCR
# structures, see the GPR68/regression work built on top of this module).
CI2_XI_BRACKET_KJ_MOL = (-0.057, -0.049)


# --------------------------------------------------------------- find_cp_peaks_and_tm --

def test_single_peak_gives_unimodal_tm():
    T = np.linspace(280.0, 360.0, 161)
    Cp = -((T - 330.0) ** 2) / 200.0  # single smooth maximum at T=330
    result = find_cp_peaks_and_tm(T, Cp)
    assert not result.is_bimodal
    assert result.tm == pytest.approx(330.0, abs=0.6)  # grid spacing (0.5 K) limits exactness
    assert result.trough_temperature is None


def test_two_peaks_gives_trough_as_tm():
    T = np.linspace(280.0, 360.0, 321)
    # Two Gaussian-like bumps at 310 and 350 K, with a real dip between them.
    Cp = np.exp(-((T - 310.0) ** 2) / 20.0) + np.exp(-((T - 350.0) ** 2) / 20.0)
    result = find_cp_peaks_and_tm(T, Cp)
    assert result.is_bimodal
    assert len(result.peak_temperatures) == 2
    assert result.peak_temperatures[0] == pytest.approx(310.0, abs=1.0)
    assert result.peak_temperatures[1] == pytest.approx(350.0, abs=1.0)
    # Trough sits between the two peaks, near the curve's midpoint minimum (~330 K).
    assert result.trough_temperature == pytest.approx(330.0, abs=3.0)
    assert result.tm == result.trough_temperature


def test_monotonic_curve_raises_calibration_error():
    T = np.linspace(280.0, 360.0, 81)
    Cp = np.linspace(0.0, 1.0, 81)  # monotonically increasing, no interior peak
    with pytest.raises(CalibrationError) as excinfo:
        find_cp_peaks_and_tm(T, Cp)
    assert excinfo.value.cp_result is not None  # Cp(T) curve attached, per Prompt 1


def test_flat_curve_raises_calibration_error():
    T = np.linspace(280.0, 360.0, 81)
    Cp = np.full(81, 0.5)
    with pytest.raises(CalibrationError):
        find_cp_peaks_and_tm(T, Cp)


def test_small_wiggle_below_prominence_is_ignored():
    T = np.linspace(280.0, 360.0, 401)
    # Dominant real peak at 340 K, plus tiny numerical-noise-scale wiggles
    # elsewhere that should not register as additional peaks.
    Cp = np.exp(-((T - 340.0) ** 2) / 50.0) + 1e-4 * np.sin(T)
    result = find_cp_peaks_and_tm(T, Cp)
    assert not result.is_bimodal
    assert result.tm == pytest.approx(340.0, abs=1.0)


def test_too_few_points_raises_value_error():
    with pytest.raises(ValueError):
        find_cp_peaks_and_tm(np.array([280.0, 281.0]), np.array([0.0, 1.0]))


# --------------------------------------------------------------- compute_delta_g_fold --

def test_compute_delta_g_fold_matches_hand_computed_windows():
    params = WSMEParams.soluble_protein_defaults()
    result = run_pipeline(CI2, ph=7.0, params=params)
    dg = compute_delta_g_fold(result.structure, result.block_model, result.ss_mask, params)

    fes = result.result.fes
    nblocks = result.block_model.nblocks
    n_values = result.result.n_values
    lo_cut = max(1, int(np.floor(0.15 * nblocks)))
    hi_cut = max(1, int(np.ceil(0.85 * nblocks)))
    expected = float(fes[n_values >= hi_cut].min()) - float(fes[n_values <= lo_cut].min())
    assert dg == pytest.approx(expected, rel=1e-6)


def test_compute_delta_g_fold_is_negative_for_a_real_stable_fold():
    # CI2 with its own native (soluble-protein) defaults should show a
    # real, stabilized fold: folded well below unfolded well.
    result = run_pipeline(CI2, ph=7.0)
    params = WSMEParams.soluble_protein_defaults()
    dg = compute_delta_g_fold(result.structure, result.block_model, result.ss_mask, params)
    assert dg < 0.0


# --------------------------------------------------------------- calibrate_xi_tm_mode --

def test_calibrate_xi_tm_mode_hits_target_and_satisfies_post_condition():
    result = run_pipeline(CI2, ph=7.0)
    params = WSMEParams.soluble_protein_defaults()
    calib = calibrate_xi_tm_mode(
        result.structure, result.block_model, result.ss_mask, params=params,
        xi_bracket_kj_mol=CI2_XI_BRACKET_KJ_MOL,
    )
    assert isinstance(calib, XiTmCalibrationResult)
    assert calib.tm_achieved_k == pytest.approx(PAPER_TARGET_TM_K, abs=2.0)
    assert calib.folded_minimum_ok is True
    assert calib.folded_minimum_frac >= 0.85
    assert calib.xi_j_mol == pytest.approx(calib.xi_kj_mol * 1000.0)
    assert calib.z_score_vs_paper == pytest.approx(
        (calib.xi_j_mol - PAPER_XI_MEAN_J_MOL) / PAPER_XI_STD_J_MOL
    )
    # Provenance carries commit + config hash even without a structure path.
    assert calib.provenance["commit"] != ""
    assert calib.provenance["config_hash"]
    assert "xi calibrated (Tm mode)" in calib.summary_header()
    assert "top 15%" in calib.summary_header() or "FAILED" in calib.summary_header()


def test_calibrate_xi_tm_mode_provenance_includes_structure_hash_when_path_given():
    result = run_pipeline(CI2, ph=7.0)
    params = WSMEParams.soluble_protein_defaults()
    calib = calibrate_xi_tm_mode(
        result.structure, result.block_model, result.ss_mask, params=params,
        xi_bracket_kj_mol=CI2_XI_BRACKET_KJ_MOL, structure_path=CI2,
    )
    assert calib.provenance["structure_sha256"] is not None
    assert len(calib.provenance["structure_sha256"]) == 64
    assert calib.provenance["structure_path"] == str(CI2)


def test_calibrate_xi_tm_mode_raises_when_bracket_has_no_root():
    # Both edges known (by direct scan) to give Tm well below 333 K --
    # no sign change in Tm(xi)-333 across this narrow, too-weak bracket.
    result = run_pipeline(CI2, ph=7.0)
    params = WSMEParams.soluble_protein_defaults()
    with pytest.raises(CalibrationError) as excinfo:
        calibrate_xi_tm_mode(
            result.structure, result.block_model, result.ss_mask, params=params,
            xi_bracket_kj_mol=(-0.052, -0.050),
        )
    assert excinfo.value.cp_result is not None


def test_folded_minimum_post_condition_helper_directly():
    from wsme_gpcr.calibration import _folded_minimum_ok

    nblocks = 20
    n_values = np.arange(1, nblocks + 1)
    # Global minimum at n=18 (90% of nblocks) -- inside the top 15%.
    fes_good = np.abs(n_values - 18).astype(float)
    good = WSMEResult(n_values=n_values, fes=fes_good, hv=10, fes2D=None, fpath=None, zfin=1.0)
    ok, n_at_min, frac = _folded_minimum_ok(good, nblocks)
    assert ok is True
    assert n_at_min == 18
    assert frac == pytest.approx(0.9)

    # Global minimum at n=15 (75% of nblocks) -- matches the bug report's
    # "profile minimum sitting near 76% structured", outside the top 15%.
    fes_bad = np.abs(n_values - 15).astype(float)
    bad = WSMEResult(n_values=n_values, fes=fes_bad, hv=10, fes2D=None, fpath=None, zfin=1.0)
    ok, n_at_min, frac = _folded_minimum_ok(bad, nblocks)
    assert ok is False
    assert frac == pytest.approx(0.75)


# --------------------------------------------------------------- calibrate_xi_isostability_mode --

def test_isostability_identity_case_recovers_the_reference_xi():
    # Reference and "other" are literally the same structure: the
    # objective function is identically zero at xi_other == xi_reference,
    # so the solver must recover (approximately) that same value.
    result = run_pipeline(CI2, ph=7.0)
    params = WSMEParams.soluble_protein_defaults()
    xi_reference_kj_mol = -0.054  # a representative value inside the validated bracket

    iso = calibrate_xi_isostability_mode(
        result.structure, result.block_model, result.ss_mask, xi_reference_kj_mol,
        result.structure, result.block_model, result.ss_mask,
        params=params, xi_bracket_kj_mol=CI2_XI_BRACKET_KJ_MOL,
    )
    assert iso.xi_other_kj_mol == pytest.approx(xi_reference_kj_mol, abs=1e-3)
    assert "IMPOSED" in iso.warning
    assert "IMPOSED" in iso.summary_header()


def test_isostability_provenance_and_common_delta_g():
    result = run_pipeline(CI2, ph=7.0)
    params = WSMEParams.soluble_protein_defaults()
    iso = calibrate_xi_isostability_mode(
        result.structure, result.block_model, result.ss_mask, -0.054,
        result.structure, result.block_model, result.ss_mask,
        params=params, xi_bracket_kj_mol=CI2_XI_BRACKET_KJ_MOL,
        reference_structure_path=CI2, other_structure_path=CI2,
    )
    assert iso.T_ref_k == 310.0
    assert iso.provenance["reference_structure_sha256"] == iso.provenance["other_structure_sha256"]
    assert iso.provenance["commit"]
    expected_dg = compute_delta_g_fold(
        result.structure, result.block_model, result.ss_mask,
        __import__("dataclasses").replace(params, ene=-0.054, T=310.0),
    )
    assert iso.delta_g_fold_common_kj_mol == pytest.approx(expected_dg, rel=1e-6)
