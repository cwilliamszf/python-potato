import numpy as np
import pytest

from linkage_pka.double_funnel import DoubleFunnelResult, build_double_funnel_landscape


def _toy_inputs(n_ph=3, nblocks_inactive=4, nblocks_active=5):
    ph_values = np.linspace(5.0, 8.0, n_ph)
    rng = np.random.default_rng(0)
    fes_inactive = rng.uniform(0.0, 50.0, size=(n_ph, nblocks_inactive))
    fes_active = rng.uniform(0.0, 50.0, size=(n_ph, nblocks_active))
    n_values_inactive = np.arange(1, nblocks_inactive + 1)
    n_values_active = np.arange(1, nblocks_active + 1)
    delta_g_activation = np.array([10.0, -5.0, -20.0])[:n_ph]
    return fes_inactive, n_values_inactive, fes_active, n_values_active, delta_g_activation, ph_values


def test_returns_dataclass_with_expected_shapes():
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)
    assert isinstance(result, DoubleFunnelResult)
    n_q = len(n_i) + len(n_a)
    assert result.q_values.shape == (n_q,)
    assert result.free_energy.shape == (len(ph), n_q)
    assert result.q_gap_index == len(n_i)


def test_q_values_span_expected_range_and_are_monotonic():
    # n_values starts at 1, not 0 (WSME's own convention -- see
    # WSMEResult.n_values), so the disordered end's fraction is 1/nblocks,
    # not exactly 0 -- the true endpoints approach but don't reach +-1.
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs(nblocks_inactive=4, nblocks_active=5)
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)
    assert result.q_values[0] == pytest.approx(1.0 / 4 - 1.0)
    assert result.q_values[-1] == pytest.approx(1.0 - 1.0 / 5)
    # Non-decreasing overall; the two reference states (one from each
    # conformer) both land exactly at Q=0, a zero-width tie at the seam --
    # not a bug, just the two real endpoints this landscape actually anchors.
    assert np.all(np.diff(result.q_values) >= 0)
    assert result.q_values[result.q_gap_index - 1] == pytest.approx(0.0)  # inactive reference, Q=0-
    assert result.q_values[result.q_gap_index] == pytest.approx(0.0)      # active reference, Q=0+


def test_inactive_reference_state_anchored_to_zero():
    # The inactive conformer's own reference structure (n=nblocks, frac=1,
    # adjacent to the seam on the inactive side) must sit at exactly 0 at
    # every pH, regardless of the raw WSME fes values fed in.
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)
    inactive_reference_col = result.q_gap_index - 1
    assert result.free_energy[:, inactive_reference_col] == pytest.approx(0.0, abs=1e-9)


def test_active_reference_state_anchored_to_delta_g_activation():
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)
    active_reference_col = result.q_gap_index
    assert result.free_energy[:, active_reference_col] == pytest.approx(dg, abs=1e-9)


def test_within_basin_shape_is_preserved_up_to_the_anchor_shift():
    # The relative depths WITHIN one conformer's WSME curve (a real,
    # uncorrupted quantity at fixed pH) must survive untouched -- only a
    # per-pH additive constant should differ from the raw input.
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)

    inactive_side = result.free_energy[:, :result.q_gap_index]  # ascending Q order = ascending n order (no reorder)
    raw_diffs = np.diff(fes_i, axis=1)
    shifted_diffs = np.diff(inactive_side, axis=1)
    assert shifted_diffs == pytest.approx(raw_diffs)

    active_side = result.free_energy[:, result.q_gap_index:]  # ascending Q order = DESCENDING n order (reversed)
    raw_diffs_active = np.diff(fes_a[:, ::-1], axis=1)
    shifted_diffs_active = np.diff(active_side, axis=1)
    assert shifted_diffs_active == pytest.approx(raw_diffs_active)


def test_disordered_ends_are_not_anchored_and_reflect_raw_shape():
    # The fully-disordered ends (Q=-1, Q=+1) are NOT the anchor point --
    # they should equal the reference-anchored value plus whatever the raw
    # WSME curve says about that far state relative to its own reference.
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)
    expected_inactive_end = fes_i[:, 0] - fes_i[:, -1]
    assert result.free_energy[:, 0] == pytest.approx(expected_inactive_end)
    expected_active_end = fes_a[:, 0] - fes_a[:, -1] + dg
    assert result.free_energy[:, -1] == pytest.approx(expected_active_end)


def test_mismatched_ph_length_raises():
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    with pytest.raises(ValueError):
        build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph[:-1])


def test_mismatched_block_axis_raises():
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    with pytest.raises(ValueError):
        build_double_funnel_landscape(fes_i, n_i[:-1], fes_a, n_a, dg, ph)


def test_too_few_blocks_raises():
    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs(nblocks_inactive=1)
    with pytest.raises(ValueError):
        build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)


def test_zero_delta_g_puts_both_reference_states_at_same_height():
    fes_i, n_i, fes_a, n_a, _, ph = _toy_inputs()
    dg_zero = np.zeros(len(ph))
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg_zero, ph)
    inactive_reference_col = result.q_gap_index - 1
    active_reference_col = result.q_gap_index
    assert result.free_energy[:, inactive_reference_col] == pytest.approx(0.0, abs=1e-9)
    assert result.free_energy[:, active_reference_col] == pytest.approx(0.0, abs=1e-9)


def test_plot_double_funnel_runs_without_error():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from linkage_pka.double_funnel import plot_double_funnel

    fes_i, n_i, fes_a, n_a, dg, ph = _toy_inputs()
    result = build_double_funnel_landscape(fes_i, n_i, fes_a, n_a, dg, ph)
    ax = plot_double_funnel(result)
    assert ax is not None
