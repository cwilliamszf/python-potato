import numpy as np
import pytest

from gpcr_energy_landscapes.energy_landscape import (
    KB_KCAL_PER_MOL_K,
    boltzmann_weights,
    free_energy_from_gibbs,
    landscape_1d,
    landscape_2d,
)


def test_boltzmann_weights_normalized_and_monotonic():
    w = boltzmann_weights(np.array([0.0, 1.0, 2.0]), temperature=310.0)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]  # lower free energy -> higher weight


def test_free_energy_from_gibbs_two_degenerate_states():
    temperature = 310.0
    rt = KB_KCAL_PER_MOL_K * temperature
    g0 = -3.5
    combined = free_energy_from_gibbs([g0, g0], temperature=temperature)
    # combining two energetically-identical microstates lowers the macrostate
    # free energy by -RT*ln(2) (extra degeneracy/entropy)
    assert combined == pytest.approx(g0 - rt * np.log(2))


def test_free_energy_from_gibbs_single_state_is_unchanged():
    assert free_energy_from_gibbs([-10.0]) == pytest.approx(-10.0)


def test_free_energy_from_gibbs_empty_is_nan():
    assert np.isnan(free_energy_from_gibbs([]))


def test_landscape_1d_kde_min_is_zero():
    rng = np.random.default_rng(0)
    cv = rng.normal(0, 1, 200)
    landscape = landscape_1d(cv, method="kde", grid_size=50)
    assert landscape["dG"].min() == pytest.approx(0.0, abs=1e-8)
    assert len(landscape["cv"]) == 50


def test_landscape_1d_histogram_bimodal_wells_are_low():
    cv = np.concatenate([np.full(50, 0.0), np.full(50, 10.0)])
    landscape = landscape_1d(cv, method="histogram", bins=20, min_count=1)
    finite_dG = landscape["dG"][np.isfinite(landscape["dG"])]
    assert finite_dG.min() == pytest.approx(0.0, abs=1e-8)
    # both equally populated basins should end up near the minimum
    low_energy_bins = finite_dG[finite_dG < 0.5]
    assert len(low_energy_bins) >= 2


def test_landscape_1d_gibbs_method_prefers_low_energy_bin():
    cv = np.array([0.0, 0.0, 10.0, 10.0])
    gibbs = np.array([-5.0, -5.0, 0.0, 0.0])
    landscape = landscape_1d(cv, gibbs=gibbs, method="histogram", bins=2, min_count=1)
    assert landscape["dG"][0] < landscape["dG"][1]


def test_landscape_2d_shapes_and_min_zero():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300)
    y = rng.normal(0, 1, 300)
    landscape = landscape_2d(x, y, method="kde", grid_size=40)
    assert landscape["dG"].shape == (40, 40)
    assert landscape["dG"].min() == pytest.approx(0.0, abs=1e-8)


def test_landscape_1d_unknown_method_raises():
    with pytest.raises(ValueError):
        landscape_1d(np.array([1.0, 2.0]), method="bogus")


def test_landscape_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        landscape_2d(np.array([1.0, 2.0]), np.array([1.0]))
