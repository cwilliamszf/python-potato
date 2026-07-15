import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpcr_ensemble import activation_state as astate


def make_chain(n=10, ca_spacing=3.8):
    """A straight synthetic CA trace with canonical bond spacing, residues 1..n."""
    return {i: np.array([i * ca_spacing, 0.0, 0.0]) for i in range(1, n + 1)}


def test_tm3_tm6_distance_basic():
    coords = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([3.0, 4.0, 0.0])}
    assert astate.tm3_tm6_distance(coords, 3, 6) == pytest.approx(5.0)


def test_tm3_tm6_distance_missing_residue_raises():
    coords = {3: np.array([0.0, 0.0, 0.0])}
    with pytest.raises(KeyError):
        astate.tm3_tm6_distance(coords, 3, 6)


def test_calibrate_thresholds_orders_and_bands_correctly():
    inactive = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([9.0, 0.0, 0.0])}
    active = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([14.0, 0.0, 0.0])}
    thresholds = astate.calibrate_thresholds(inactive, active, 3, 6, margin_fraction=0.15)
    assert thresholds.inactive_max < thresholds.active_min
    assert thresholds.inactive_max > 9.0
    assert thresholds.active_min < 14.0


def test_calibrate_thresholds_rejects_inverted_refs():
    inactive = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([14.0, 0.0, 0.0])}
    active = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([9.0, 0.0, 0.0])}
    with pytest.raises(ValueError):
        astate.calibrate_thresholds(inactive, active, 3, 6)


def test_classify_model_three_states():
    thresholds = astate.ActivationThresholds(inactive_max=10.0, active_min=13.0)
    inactive_coords = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([8.0, 0.0, 0.0])}
    active_coords = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([15.0, 0.0, 0.0])}
    intermediate_coords = {3: np.array([0.0, 0.0, 0.0]), 6: np.array([11.5, 0.0, 0.0])}

    label, d = astate.classify_model(inactive_coords, 3, 6, thresholds)
    assert label == "inactive"
    label, d = astate.classify_model(active_coords, 3, 6, thresholds)
    assert label == "active"
    label, d = astate.classify_model(intermediate_coords, 3, 6, thresholds)
    assert label == "intermediate"


def test_passes_fold_quality_accepts_well_formed_chain():
    coords = make_chain()
    assert astate.passes_fold_quality(coords, mean_plddt_value=85.0, plddt_cutoff=70.0)


def test_passes_fold_quality_rejects_low_plddt():
    coords = make_chain()
    assert not astate.passes_fold_quality(coords, mean_plddt_value=40.0, plddt_cutoff=70.0)


def test_passes_fold_quality_rejects_broken_geometry():
    coords = make_chain()
    coords[5] = coords[5] + np.array([20.0, 0.0, 0.0])  # blow up one CA-CA bond
    assert not astate.passes_fold_quality(coords, mean_plddt_value=90.0, plddt_cutoff=70.0)
