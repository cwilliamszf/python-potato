import shutil
from pathlib import Path

import numpy as np
import pytest

from wsme_gpcr.blocking import BlockModel
from wsme_gpcr.calibration import (
    CalibrationError,
    PAPER_TARGET_TM_K,
    PAPER_XI_MEAN_J_MOL,
    PAPER_XI_STD_J_MOL,
    XiTmCalibrationResult,
    XiFoldScanResult,
    calibrate_xi_isostability_mode,
    calibrate_xi_tm_mode,
    compute_delta_g_fold,
    compute_fc,
    find_cp_peaks_and_tm,
    xi_fold_scan,
)
from wsme_gpcr.coupling import CouplingResult
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


# --------------------------------------------------------------- compute_fc --

def _synthetic_block_model(residues_per_block):
    """residues_per_block: list of residue counts per block, e.g. [4,4,3]."""
    nblocks = len(residues_per_block)
    nres = sum(residues_per_block)
    block_of_residue = np.repeat(np.arange(nblocks), residues_per_block)
    block_residue_range = np.zeros((nblocks, 2), dtype=int)
    pos = 0
    for b, n in enumerate(residues_per_block):
        block_residue_range[b] = [pos, pos + n - 1]
        pos += n
    return BlockModel(
        nres=nres, nblocks=nblocks, block_size=4, block_of_residue=block_of_residue,
        block_residue_range=block_residue_range, block_cmap=np.zeros((nblocks, nblocks)),
        block_elec=np.zeros((0, 5)),
    )


def _synthetic_coupling(cfe: np.ndarray) -> CouplingResult:
    nb = cfe.shape[0]
    nan = np.full((nb, nb), np.nan)
    return CouplingResult(
        p_folded=np.zeros(nb), p_folded_folded=nan, p_folded_unfolded=nan, p_unfolded_unfolded=nan,
        chi_plus=nan, chi_minus=nan, coupling_free_energy=cfe, zfin=1.0,
    )


def test_compute_fc_counts_residues_in_z_scored_strongly_coupled_blocks():
    # 4 blocks, sizes [3, 3, 2, 2] (10 residues total). Row-means (each
    # row's off-diagonal entries are a single repeated value, so nanmean
    # is exactly that value) are [20, 5, 5, 5] -- an asymmetric 1-vs-3
    # split, hand-verified: mean=8.75, population std=6.495,
    # Z_block0=(20-8.75)/6.495=+1.73 (>1, qualifies),
    # Z_others=(5-8.75)/6.495=-0.58 (does not qualify).
    block_model = _synthetic_block_model([3, 3, 2, 2])
    cfe = np.array([
        [np.nan, 20.0, 20.0, 20.0],
        [5.0, np.nan, 5.0, 5.0],
        [5.0, 5.0, np.nan, 5.0],
        [5.0, 5.0, 5.0, np.nan],
    ])
    coupling = _synthetic_coupling(cfe)
    fc = compute_fc(coupling, block_model)
    # Only block 0 (3 residues) qualifies.
    assert fc == pytest.approx(3.0 / 10.0)


def test_compute_fc_all_nan_gives_zero():
    block_model = _synthetic_block_model([4, 4, 2])
    cfe = np.full((3, 3), np.nan)
    coupling = _synthetic_coupling(cfe)
    fc = compute_fc(coupling, block_model)
    assert fc == 0.0


def test_compute_fc_uniform_coupling_gives_zero_not_one():
    # Every block equally coupled -> zero variance in the row-mean
    # population -> Z-score is undefined (not "everyone qualifies"), so
    # this must give 0.0, not 1.0. A fixed-threshold implementation would
    # have called this "all strongly coupled"; the paper's real Z-score
    # procedure correctly calls it "no relative differentiation possible."
    block_model = _synthetic_block_model([4, 4, 2])
    cfe = np.full((3, 3), 10.0)
    np.fill_diagonal(cfe, np.nan)
    coupling = _synthetic_coupling(cfe)
    fc = compute_fc(coupling, block_model)
    assert fc == 0.0


def test_compute_fc_z_threshold_is_tunable():
    block_model = _synthetic_block_model([3, 3, 2, 2])
    cfe = np.array([
        [np.nan, 20.0, 20.0, 20.0],
        [5.0, np.nan, 5.0, 5.0],
        [5.0, 5.0, np.nan, 5.0],
        [5.0, 5.0, 5.0, np.nan],
    ])
    coupling = _synthetic_coupling(cfe)
    # Block 0's Z-score is +1.73 -- qualifies at 1.0, not at 2.0.
    assert compute_fc(coupling, block_model, z_threshold=1.0) == pytest.approx(3.0 / 10.0)
    assert compute_fc(coupling, block_model, z_threshold=2.0) == pytest.approx(0.0)


def test_compute_fc_real_plumbing_on_ci2():
    result = run_pipeline(CI2, ph=7.0, with_coupling=True)
    fc = compute_fc(result.coupling_result, result.block_model)
    assert 0.0 <= fc <= 1.0


# fc validation against the paper's own real per-receptor ground truth.
# Values below are DeltaGc_310_<tag>-derived fc (Z-score>1 fraction),
# computed directly from the paper's own bundled CouplingMat_310/
# DeltaGc_310 in the GPCR-Landscapes reference repo's .mat files (not
# re-derived from this pipeline) -- see FINDINGS.md's "fc definition
# corrected" entry for the full derivation and cross-checks. This is the
# fidelity gate examples/calibration_regression.py's Tier 2 depends on;
# the tolerance (7 percentage points) reflects the measured real spread
# (max observed diff 5.4pp across 5 receptors) from residual block-
# partition differences (DSSP vs. the paper's own STRIDE -- see the
# block-partition audit), not a loosened/tuned-to-pass margin.
_REAL_FC_CASES = [
    ("gpcr9i", -0.0499, 12.15),
    ("gpcr1i", -0.0482, 9.77),
    ("gpcr20i", -0.0452, 15.10),
    ("gpcr13a", -0.0550, 15.95),
    ("gpcr2i", -0.0563, 9.46),
]
_REFDIR = Path(__file__).resolve().parent.parent / "examples" / "data" / "gpcr_landscapes_reference"


@pytest.mark.skipif(shutil.which("mkdssp") is None and shutil.which("dssp") is None,
                     reason="requires mkdssp for paper-fidelity blocking")
@pytest.mark.parametrize("tag,ene,paper_fc_pct", _REAL_FC_CASES)
def test_compute_fc_matches_paper_real_per_receptor_fc(tag, ene, paper_fc_pct):
    from wsme_gpcr.pipeline import run_pipeline as _run_pipeline
    from wsme_gpcr.wsme import WSMEParams
    from wsme_gpcr.coupling import compute_coupling

    r = _run_pipeline(str(_REFDIR / f"{tag}.pdb"), ph=7.0, use_dssp=True)
    params = WSMEParams(T=310.0, ene=ene)
    cr = compute_coupling(r.structure, r.block_model, r.ss_mask, params)
    fc_pct = compute_fc(cr, r.block_model) * 100
    assert fc_pct == pytest.approx(paper_fc_pct, abs=7.0)


# --------------------------------------------------------------- xi_fold_scan --

def test_xi_fold_scan_real_plumbing_ci2():
    result = run_pipeline(CI2, ph=7.0)
    params = WSMEParams.soluble_protein_defaults()
    scan = xi_fold_scan(result.structure, result.block_model, result.ss_mask, params,
                         xi_range_j_mol=(-100.0, -90.0), step_j_mol=5.0)
    assert isinstance(scan, XiFoldScanResult)
    assert scan.xi_values_j_mol.tolist() == [-100.0, -95.0, -90.0]
    assert len(scan.fold_fracs) == 3
    # CI2 folds robustly across this whole strongly-stabilizing range.
    assert scan.folds_anywhere is True
    assert scan.best_fold_frac == pytest.approx(max(scan.fold_fracs))


def test_xi_fold_scan_detects_a_sharp_synthetic_transition(monkeypatch):
    # Force run_wsme to return a fold fraction that flips sharply at a
    # known xi, so n_transitions/folds_anywhere can be checked exactly
    # without depending on a real structure's own transition location.
    import wsme_gpcr.calibration as calib_mod

    class FakeResult:
        def __init__(self, n_values, fes):
            self.n_values = n_values
            self.fes = fes

    def fake_run_wsme(structure, block_model, ss_mask, params):
        nblocks = 10
        n_values = np.arange(1, nblocks + 1)
        # folded (n=9) is the minimum only when ene <= -50 J/mol; otherwise n=1 wins.
        fes = np.full(nblocks, 100.0)
        if params.ene * 1000 <= -50.0:
            fes[8] = 0.0
        else:
            fes[0] = 0.0
        return FakeResult(n_values, fes)

    monkeypatch.setattr(calib_mod, "run_wsme", fake_run_wsme)

    class FakeBlockModel:
        nblocks = 10

    scan = xi_fold_scan(None, FakeBlockModel(), None, WSMEParams(),
                         xi_range_j_mol=(-55.0, -45.0), step_j_mol=1.0)
    assert scan.folds_anywhere is True
    assert scan.best_xi_j_mol <= -50.0
    assert scan.best_fold_frac == pytest.approx(0.9)
    assert scan.n_transitions == 1  # exactly one flip, at the -50 boundary


def test_xi_fold_scan_folds_anywhere_false_when_never_folds(monkeypatch):
    import wsme_gpcr.calibration as calib_mod

    class FakeResult:
        def __init__(self, n_values, fes):
            self.n_values = n_values
            self.fes = fes

    def fake_run_wsme(structure, block_model, ss_mask, params):
        nblocks = 10
        n_values = np.arange(1, nblocks + 1)
        fes = np.full(nblocks, 100.0)
        fes[0] = 0.0  # always fully unfolded, regardless of ene
        return FakeResult(n_values, fes)

    monkeypatch.setattr(calib_mod, "run_wsme", fake_run_wsme)

    class FakeBlockModel:
        nblocks = 10

    scan = xi_fold_scan(None, FakeBlockModel(), None, WSMEParams(),
                         xi_range_j_mol=(-55.0, -45.0), step_j_mol=1.0)
    assert scan.folds_anywhere is False
    assert scan.n_transitions == 0
