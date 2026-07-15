import numpy as np
import pytest

from linkage_pka.linkage import compute_linkage, protonation_fraction, sensitivity_band, LN10


def test_protonation_fraction_is_half_at_pka():
    assert protonation_fraction(7.0, 7.0) == pytest.approx(0.5)


def test_protonation_fraction_monotonic_decreasing_in_ph():
    ph = np.linspace(2, 12, 50)
    theta = protonation_fraction(ph, 7.0)
    assert np.all(np.diff(theta) <= 0)
    assert theta[0] == pytest.approx(1.0, abs=1e-4)   # far below pKa: fully protonated
    assert theta[-1] == pytest.approx(0.0, abs=1e-4)  # far above pKa: fully deprotonated


def test_identical_pka_gives_zero_linkage():
    ph = np.linspace(5, 8, 10)
    pka = {10: 4.0, 20: 6.5, 30: 9.0}
    result = compute_linkage(ph, pka_active=pka, pka_inactive=pka)
    assert np.allclose(result.delta_g_act, 0.0, atol=1e-10)
    assert np.allclose(result.delta_n_h, 0.0, atol=1e-10)


def test_single_site_matches_hand_calculation():
    # One site, pKa shifts from 6.0 (inactive) to 8.0 (active) -- a classic
    # "upshifted buried carboxylate favors the active state at low pH" case.
    ph = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    result = compute_linkage(ph, pka_active={1: 8.0}, pka_inactive={1: 6.0})

    RT = 8.31446261815324e-3 * 298.15
    expected_dg = -RT * (np.log(1 + 10 ** (8.0 - ph)) - np.log(1 + 10 ** (6.0 - ph)))
    assert np.allclose(result.delta_g_act, expected_dg, rtol=1e-8)

    theta_active = 1 / (1 + 10 ** (ph - 8.0))
    theta_inactive = 1 / (1 + 10 ** (ph - 6.0))
    assert np.allclose(result.delta_n_h, theta_active - theta_inactive, rtol=1e-8)

    # At low pH both sites are protonated (theta~1) -> delta_n_h ~ 0 (both
    # already saturated); the interesting uptake happens between the two pKa's.
    assert result.delta_n_h[np.argmin(np.abs(ph - 7.0))] > 0.3


def test_wyman_relation_holds_numerically():
    """The core physics check: Delta_n_H(pH) must equal
    d(-DeltaG_act(pH)/RT) / d(ln[H+]) via finite differences on
    delta_g_act alone -- i.e. delta_n_h is not an independently-invented
    formula, it is *the derivative of* delta_g_act, exactly as Wyman's
    linked-function theorem requires. This validates the theta(pH)
    uniform-fraction-protonated convention documented in linkage.py's
    module docstring against the (independently, structurally standard)
    delta_g_act formula."""
    T = 298.15
    RT = 8.31446261815324e-3 * T
    pka_active = {1: 7.4, 2: 5.1, 3: 10.2}
    pka_inactive = {1: 6.0, 2: 5.1, 3: 9.0}  # site 2 unchanged -> must contribute 0

    ph0 = 6.7
    h = 1e-5
    ph_grid = np.array([ph0 - h, ph0, ph0 + h])
    result = compute_linkage(ph_grid, pka_active, pka_inactive, T=T)

    # d(pH)/d(ln[H+]) = -1/ln(10); lnK = -DeltaG_act/RT
    dlnK_dph = (-result.delta_g_act[2] / RT - (-result.delta_g_act[0] / RT)) / (2 * h)
    dlnK_dlnH = dlnK_dph * (-1.0 / LN10)

    assert dlnK_dlnH == pytest.approx(result.delta_n_h[1], rel=1e-4)

    # site 2 (unchanged pKa) must not contribute
    site2_idx = list(result.resnums).index(2)
    assert result.delta_n_h_per_residue[1, site2_idx] == pytest.approx(0.0, abs=1e-10)


def test_per_residue_decomposition_sums_to_total():
    ph = np.linspace(5, 8, 7)
    pka_active = {1: 7.0, 2: 4.5, 3: 9.0}
    pka_inactive = {1: 5.5, 2: 4.5, 3: 9.5}
    result = compute_linkage(ph, pka_active, pka_inactive)
    assert np.allclose(result.delta_n_h, result.delta_n_h_per_residue.sum(axis=1), rtol=1e-10)


def test_top_contributors_ranks_by_magnitude():
    ph = np.array([6.5])
    pka_active = {1: 7.0, 2: 4.5, 3: 9.0}
    pka_inactive = {1: 5.0, 2: 4.5, 3: 9.5}  # site 1 has the largest pKa shift near pH 6.5
    result = compute_linkage(ph, pka_active, pka_inactive)
    top = result.top_contributors(6.5, n=3)
    assert top[0][0] == 1  # largest |contribution| first
    resnums_ranked = [r for r, _ in top]
    assert set(resnums_ranked) == {1, 2, 3}


def test_only_sites_in_both_states_are_used():
    ph = np.array([7.0])
    result = compute_linkage(ph, pka_active={1: 7.0, 2: 5.0}, pka_inactive={1: 6.0, 3: 8.0})
    assert list(result.resnums) == [1]  # site 2 (active-only) and 3 (inactive-only) dropped


def test_missing_pka_value_propagates_as_nan_not_dropped_silently():
    ph = np.array([7.0])
    result = compute_linkage(ph, pka_active={1: 7.0, 2: None}, pka_inactive={1: 6.0, 2: 5.0})
    idx2 = list(result.resnums).index(2)
    assert np.isnan(result.delta_n_h_per_residue[0, idx2])
    # site 1 (fully resolved) must still be summed correctly despite site 2's NaN
    assert np.isfinite(result.delta_n_h[0])


def test_no_shared_residues_raises():
    with pytest.raises(ValueError):
        compute_linkage([7.0], pka_active={1: 7.0}, pka_inactive={2: 7.0})


def test_sensitivity_band_reports_spread_across_variants():
    ph = np.linspace(5, 8, 4)
    with_ion = compute_linkage(ph, pka_active={1: 7.0}, pka_inactive={1: 5.0})
    without_ion = compute_linkage(ph, pka_active={1: 7.5}, pka_inactive={1: 5.0})
    band = sensitivity_band([with_ion, without_ion])
    assert np.all(band["spread"] >= 0)
    assert np.allclose(band["max"] - band["min"], band["spread"])
