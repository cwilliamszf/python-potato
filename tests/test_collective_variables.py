import numpy as np
import pytest

from gpcr_energy_landscapes.collective_variables import (
    atom_distance,
    closest_heavy_atom_distance,
    connector_delta_rmsd,
    evaluate_cv,
    rmsd,
)
from tests.helpers import make_structure


def test_closest_heavy_atom_distance():
    structure = make_structure(
        "s1",
        "A",
        {
            207: {"CA": (0, 0, 0), "CB": (1, 0, 0)},
            315: {"CA": (0, 0, 3), "CB": (0, 0, 2)},
        },
    )
    d = closest_heavy_atom_distance(structure, {"chain": "A", "resid": 207}, {"chain": "A", "resid": 315})
    # closest pair is CA(207) <-> CB(315), distance 2.0
    assert d == pytest.approx(2.0)


def test_atom_distance():
    structure = make_structure(
        "s1",
        "A",
        {
            219: {"CZ": (0, 0, 0)},
            326: {"CZ": (3, 4, 0)},
        },
    )
    d = atom_distance(structure, {"chain": "A", "resid": 219, "atom": "CZ"}, {"chain": "A", "resid": 326, "atom": "CZ"})
    assert d == pytest.approx(5.0)


def test_rmsd():
    a = np.array([[0, 0, 0], [1, 0, 0]])
    b = np.array([[0, 0, 0], [1, 0, 1]])
    assert rmsd(a, b) == pytest.approx(np.sqrt(0.5))


def test_rmsd_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmsd(np.zeros((2, 3)), np.zeros((3, 3)))


def test_connector_delta_rmsd_prefers_closer_reference():
    query = make_structure("query", "A", {121: {"CA": (0, 0, 0)}, 282: {"CA": (1, 0, 0)}})
    active = make_structure("active", "A", {121: {"CA": (0, 0, 0)}, 282: {"CA": (1, 0, 0)}})
    inactive = make_structure("inactive", "A", {121: {"CA": (0, 0, 5)}, 282: {"CA": (1, 0, 5)}})

    d_rmsd = connector_delta_rmsd(query, active, inactive, residues=[121, 282], chain="A", atom_names=["CA"])
    # identical to active (rmsd=0), 5 Angstrom from inactive along z -> delta = 0 - 5
    assert d_rmsd == pytest.approx(-5.0)


def test_evaluate_cv_dispatches_by_type():
    structure = make_structure("s1", "A", {207: {"CA": (0, 0, 0)}, 315: {"CA": (0, 0, 2)}})
    cv_def = {"name": "d", "type": "closest_heavy_distance", "sel1": {"chain": "A", "resid": 207}, "sel2": {"chain": "A", "resid": 315}}
    assert evaluate_cv(structure, cv_def) == pytest.approx(2.0)


def test_evaluate_cv_unknown_type_raises():
    structure = make_structure("s1", "A", {1: {"CA": (0, 0, 0)}})
    with pytest.raises(ValueError):
        evaluate_cv(structure, {"name": "x", "type": "not_a_real_type"})
