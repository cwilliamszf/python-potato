import pytest

from linkage_pka.gate_a import (
    GATE_A_RMSE_THRESHOLD_PKA_UNITS,
    SNASE_1STN_EXPERIMENTAL_PKA,
    ExperimentalPka,
    GateAResult,
    compute_gate_a_rmse,
)


def test_dataset_has_24_entries():
    assert len(SNASE_1STN_EXPERIMENTAL_PKA) == 24


def test_dataset_residue_type_counts():
    counts = {}
    for e in SNASE_1STN_EXPERIMENTAL_PKA:
        counts[e.resname] = counts.get(e.resname, 0) + 1
    assert counts == {"HIS": 4, "ASP": 8, "GLU": 12}


def test_biphasic_entries_parsed_correctly():
    by_resnum = {e.resnum: e for e in SNASE_1STN_EXPERIMENTAL_PKA}
    asp19 = by_resnum[19]
    assert asp19.expt_pka is None
    assert asp19.expt_pka_biphasic == (2.21, 6.54)
    assert asp19.is_upper_bound is False

    asp21 = by_resnum[21]
    assert asp21.expt_pka_biphasic == (3.01, 6.54)


def test_upper_bound_entries_parsed_correctly():
    by_resnum = {e.resnum: e for e in SNASE_1STN_EXPERIMENTAL_PKA}
    asp77 = by_resnum[77]
    assert asp77.is_upper_bound is True
    assert asp77.expt_pka == 2.2
    assert asp77.expt_pka_biphasic is None

    asp83 = by_resnum[83]
    assert asp83.is_upper_bound is True
    assert asp83.expt_pka == 2.2


def test_plain_entry_parsed_correctly():
    by_resnum = {e.resnum: e for e in SNASE_1STN_EXPERIMENTAL_PKA}
    his8 = by_resnum[8]
    assert his8.expt_pka == 6.52
    assert his8.expt_pka_biphasic is None
    assert his8.is_upper_bound is False
    assert his8.sasa_percent == 71.1


def test_compute_gate_a_rmse_perfect_match_gives_zero():
    computed = {e.resnum: (e.expt_pka if e.expt_pka is not None else e.expt_pka_biphasic[0])
                for e in SNASE_1STN_EXPERIMENTAL_PKA}
    result = compute_gate_a_rmse(computed)
    assert result.rmse == pytest.approx(0.0, abs=1e-9)
    assert result.mae == pytest.approx(0.0, abs=1e-9)
    assert result.passed is True


def test_compute_gate_a_rmse_excludes_biphasic_by_default():
    computed = {e.resnum: 5.0 for e in SNASE_1STN_EXPERIMENTAL_PKA}
    result = compute_gate_a_rmse(computed)
    skipped_resnums = {r for r, _, _ in result.skipped}
    assert 19 in skipped_resnums
    assert 21 in skipped_resnums
    assert all(resnum not in (19, 21) for resnum, _, _, _, _ in result.per_residue)


def test_compute_gate_a_rmse_excludes_upper_bounds_by_default():
    computed = {e.resnum: 5.0 for e in SNASE_1STN_EXPERIMENTAL_PKA}
    result = compute_gate_a_rmse(computed)
    skipped_resnums = {r for r, _, _ in result.skipped}
    assert 77 in skipped_resnums
    assert 83 in skipped_resnums


def test_compute_gate_a_rmse_can_include_upper_bounds():
    computed = {e.resnum: e.expt_pka if e.expt_pka is not None else 5.0
                for e in SNASE_1STN_EXPERIMENTAL_PKA}
    result_default = compute_gate_a_rmse(computed)
    result_included = compute_gate_a_rmse(computed, include_upper_bounds=True)
    assert result_included.n_compared == result_default.n_compared + 2
    included_resnums = {r for r, *_ in result_included.per_residue}
    assert 77 in included_resnums
    assert 83 in included_resnums


def test_compute_gate_a_rmse_skips_unmatched_resnums():
    # Only supply a computed pKa for a handful of sites -- the rest should
    # land in `skipped`, not silently vanish.
    computed = {8: 6.5, 46: 5.9}
    result = compute_gate_a_rmse(computed)
    assert result.n_compared == 2
    skipped_reasons = {r: reason for r, _, reason in result.skipped}
    assert "no computed pKa supplied" in skipped_reasons[40]


def test_compute_gate_a_rmse_raises_on_zero_overlap():
    with pytest.raises(ValueError):
        compute_gate_a_rmse({9999: 5.0})


def test_compute_gate_a_rmse_passed_flag_respects_threshold():
    computed = {e.resnum: (e.expt_pka if e.expt_pka is not None else e.expt_pka_biphasic[0]) + 2.0
                for e in SNASE_1STN_EXPERIMENTAL_PKA}
    result = compute_gate_a_rmse(computed)
    assert result.rmse == pytest.approx(2.0, abs=1e-9)
    assert result.passed is False
    assert GATE_A_RMSE_THRESHOLD_PKA_UNITS == 1.0


def test_compute_gate_a_rmse_custom_entries_and_threshold():
    entries = [
        ExperimentalPka(resnum=1, resname="ASP", expt_pka=3.0, expt_pka_biphasic=None,
                         is_upper_bound=False, expt_uncertainty="0.1", sasa_percent=10.0,
                         method="NMR", reference="test"),
    ]
    result = compute_gate_a_rmse({1: 3.5}, entries=entries, threshold_pka_units=0.4)
    assert result.n_compared == 1
    assert result.rmse == pytest.approx(0.5, abs=1e-9)
    assert result.passed is False
