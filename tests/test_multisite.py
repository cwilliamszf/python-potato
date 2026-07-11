import numpy as np
import pytest

from linkage_pka.linkage import (
    R_KJ_PER_MOL_K,
    compute_linkage,
    delta_g_act_from_ln_z,
    delta_n_h_from_theta,
    protonation_fraction,
)
from linkage_pka.multisite import (
    MAX_EXACT_CLUSTER_SIZE,
    cluster_sites,
    solve_cluster_titration,
    solve_titration,
)

RT = R_KJ_PER_MOL_K * 298.15


def test_cluster_sites_no_coupling_gives_all_singletons():
    clusters = cluster_sites([1, 2, 3], coupling={})
    assert sorted(clusters) == [[1], [2], [3]]


def test_cluster_sites_below_threshold_stays_separate():
    coupling = {(1, 2): 1.0}  # below default 2.5 kJ/mol threshold
    clusters = cluster_sites([1, 2], coupling)
    assert sorted(clusters) == [[1], [2]]


def test_cluster_sites_above_threshold_groups_together():
    coupling = {(1, 2): 5.0}
    clusters = cluster_sites([1, 2, 3], coupling)
    assert sorted(clusters) == [[1, 2], [3]]


def test_cluster_sites_transitive_chain():
    # A-B coupled, B-C coupled, A-C not directly coupled -> still one
    # connected component via B.
    coupling = {(1, 2): 5.0, (2, 3): 5.0}
    clusters = cluster_sites([1, 2, 3], coupling)
    assert clusters == [[1, 2, 3]]


def test_cluster_sites_accepts_either_key_order():
    coupling = {(2, 1): 5.0}
    clusters = cluster_sites([1, 2], coupling)
    assert clusters == [[1, 2]]


def test_singleton_cluster_matches_henderson_hasselbalch():
    ph = np.linspace(4, 10, 25)
    result = solve_cluster_titration({1: 7.0}, coupling={}, cluster_resnums=[1], ph_values=ph)
    expected = protonation_fraction(ph, 7.0)
    assert np.allclose(result.theta[1], expected, rtol=1e-10, atol=1e-12)


def test_singleton_cluster_ln_z_matches_closed_form():
    # For an isolated site, Z = 1 + 10^(pKa-pH) exactly -- the same
    # closed form used inside linkage.compute_linkage's _log1p_10pow.
    ph = np.linspace(3, 11, 17)
    result = solve_cluster_titration({5: 6.3}, coupling={}, cluster_resnums=[5], ph_values=ph)
    expected_ln_z = np.log1p(10.0 ** (6.3 - ph))
    assert np.allclose(result.ln_z, expected_ln_z, rtol=1e-8)


def test_uncoupled_pair_cluster_theta_matches_independent_hh():
    # Two sites in the same "cluster" call but with zero coupling between
    # them must reproduce independent HH occupancies exactly.
    ph = np.linspace(4, 10, 13)
    result = solve_cluster_titration(
        {1: 6.0, 2: 8.0}, coupling={(1, 2): 0.0}, cluster_resnums=[1, 2], ph_values=ph
    )
    assert np.allclose(result.theta[1], protonation_fraction(ph, 6.0), rtol=1e-8)
    assert np.allclose(result.theta[2], protonation_fraction(ph, 8.0), rtol=1e-8)


def test_two_site_coupling_matches_hand_calculation_at_matched_pka():
    # pKa_i = pKa_j = pH = 7.0, strong repulsive (anti-cooperative) coupling
    # W_ij = 20 kJ/mol: double-protonation is strongly disfavored, so each
    # site's marginal theta must drop below the uncoupled value of 0.5.
    ph = np.array([7.0])
    w = 20.0
    result = solve_cluster_titration(
        {1: 7.0, 2: 7.0}, coupling={(1, 2): w}, cluster_resnums=[1, 2], ph_values=ph
    )
    # G(00)=0, G(10)=0, G(01)=0, G(11)=w -- hand-computed partition function.
    z = 1.0 + 1.0 + 1.0 + np.exp(-w / RT)
    expected_theta = (1.0 + np.exp(-w / RT)) / z
    assert result.theta[1][0] == pytest.approx(expected_theta, rel=1e-8)
    assert result.theta[2][0] == pytest.approx(expected_theta, rel=1e-8)
    assert expected_theta < 0.5  # anti-cooperativity suppresses joint occupancy
    assert result.ln_z[0] == pytest.approx(np.log(z), rel=1e-8)


def test_two_site_coupling_symmetric_sites_give_equal_theta():
    ph = np.linspace(5, 9, 9)
    result = solve_cluster_titration(
        {10: 7.2, 20: 7.2}, coupling={(10, 20): 6.0}, cluster_resnums=[10, 20], ph_values=ph
    )
    assert np.allclose(result.theta[10], result.theta[20], rtol=1e-10)


def test_theta_bounded_in_unit_interval():
    ph = np.linspace(2, 12, 21)
    result = solve_cluster_titration(
        {1: 6.0, 2: 7.5, 3: 9.0},
        coupling={(1, 2): 8.0, (2, 3): -5.0, (1, 3): 3.0},
        cluster_resnums=[1, 2, 3],
        ph_values=ph,
    )
    for arr in result.theta.values():
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)


def test_oversized_cluster_raises_instead_of_approximating():
    n = MAX_EXACT_CLUSTER_SIZE + 1
    resnums = list(range(n))
    pka = {r: 7.0 for r in resnums}
    with pytest.raises(ValueError):
        solve_cluster_titration(pka, coupling={}, cluster_resnums=resnums, ph_values=[7.0])


def test_solve_titration_covers_every_site():
    ph = np.array([6.0, 7.0])
    pka = {1: 6.0, 2: 7.0, 3: 8.0}
    coupling = {(1, 2): 5.0}
    result = solve_titration(pka, coupling, ph)
    assert set(result.theta) == {1, 2, 3}
    assert sorted(map(tuple, result.clusters)) == [(1, 2), (3,)]


def test_solve_titration_ln_z_total_is_additive_across_clusters():
    ph = np.linspace(4, 10, 7)
    pka = {1: 6.0, 2: 6.0, 3: 8.5}
    coupling = {(1, 2): 6.0}  # sites 1,2 clustered together; 3 stands alone
    result = solve_titration(pka, coupling, ph)

    cluster12 = solve_cluster_titration({1: 6.0, 2: 6.0}, coupling, [1, 2], ph)
    cluster3 = solve_cluster_titration({3: 8.5}, {}, [3], ph)
    expected_total = cluster12.ln_z + cluster3.ln_z
    assert np.allclose(result.ln_z_total, expected_total, rtol=1e-10)


def test_uncoupled_multisite_theta_matches_compute_linkage_delta_n_h():
    # Cross-check against the independent-site closed form in linkage.py:
    # with zero coupling, the multi-site solver's theta(pH) must give the
    # exact same Delta_n_H(pH) as compute_linkage's HH-based calculation.
    ph = np.linspace(5, 9, 11)
    pka_active = {1: 7.4, 2: 5.1, 3: 10.2}
    pka_inactive = {1: 6.0, 2: 5.1, 3: 9.0}

    active = solve_titration(pka_active, coupling={}, ph_values=ph)
    inactive = solve_titration(pka_inactive, coupling={}, ph_values=ph)

    resnums, per_residue, delta_n_h = delta_n_h_from_theta(active.theta, inactive.theta)

    reference = compute_linkage(ph, pka_active, pka_inactive)
    # Align residue order (delta_n_h_from_theta sorts resnums; compute_linkage does too).
    assert list(resnums) == list(reference.resnums)
    assert np.allclose(delta_n_h, reference.delta_n_h, rtol=1e-8, atol=1e-10)
    assert np.allclose(per_residue, reference.delta_n_h_per_residue, rtol=1e-8, atol=1e-10)


def test_uncoupled_multisite_ln_z_matches_compute_linkage_delta_g_act():
    ph = np.linspace(5, 9, 11)
    pka_active = {1: 7.4, 2: 5.1, 3: 10.2}
    pka_inactive = {1: 6.0, 2: 5.1, 3: 9.0}

    active = solve_titration(pka_active, coupling={}, ph_values=ph)
    inactive = solve_titration(pka_inactive, coupling={}, ph_values=ph)

    delta_g = delta_g_act_from_ln_z(active.ln_z_total, inactive.ln_z_total)
    reference = compute_linkage(ph, pka_active, pka_inactive)
    assert np.allclose(delta_g, reference.delta_g_act, rtol=1e-8, atol=1e-8)


def test_coupling_changes_delta_g_act_relative_to_uncoupled_case():
    # Sanity check that coupling actually does something: introducing a
    # nonzero W_ij between two sites must change ln_z_total (and hence
    # DeltaDeltaG_act) relative to the coupling={} case at a pH where both
    # sites are partially occupied.
    ph = np.array([7.0])
    pka = {1: 7.0, 2: 7.0}

    uncoupled = solve_titration(pka, coupling={}, ph_values=ph)
    coupled = solve_titration(pka, coupling={(1, 2): 15.0}, ph_values=ph)

    assert not np.allclose(uncoupled.ln_z_total, coupled.ln_z_total)


def test_cluster_resnums_dict_keys_are_author_resnums_not_indices():
    # theta dict must be keyed by the actual resnum values passed in, not
    # by 0-based cluster-local indices.
    ph = np.array([7.0])
    result = solve_cluster_titration(
        {101: 6.5, 205: 8.0}, coupling={(101, 205): 4.0}, cluster_resnums=[101, 205], ph_values=ph
    )
    assert set(result.theta) == {101, 205}
