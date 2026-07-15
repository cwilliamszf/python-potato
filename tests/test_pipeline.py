import numpy as np
import pandas as pd
import pytest

from gpcr_energy_landscapes import pipeline
from gpcr_energy_landscapes.collective_variables import BETA2AR_MICROSWITCHES
from tests.helpers import make_structure


def _make_conformer(conformer_id, tm5_gap, ionic_gap, yy_gap, connector_shift):
    residues = {
        207: {"CA": (0, 0, 0)},
        315: {"CA": (0, 0, tm5_gap)},
        268: {"CA": (10, 0, 0)},
        131: {"CA": (10, 0, ionic_gap)},
        219: {"CZ": (20, 0, 0)},
        326: {"CZ": (20, 0, yy_gap)},
        121: {"CA": (0, 0, connector_shift)},
        282: {"CA": (1, 0, connector_shift)},
    }
    return make_structure(conformer_id, "A", residues)


@pytest.fixture
def ensemble_and_energies():
    # tm5_gap and ionic_gap are deliberately *not* proportional to each other
    # so the two CVs aren't perfectly collinear (a 2D KDE needs a non-singular
    # covariance, which real, noisy ensembles satisfy but a naive synthetic
    # fixture might not).
    ensemble = {
        "conf_active": _make_conformer("conf_active", tm5_gap=1.0, ionic_gap=1.5, yy_gap=1.0, connector_shift=0.0),
        "conf_inactive": _make_conformer("conf_inactive", tm5_gap=3.0, ionic_gap=2.0, yy_gap=3.0, connector_shift=5.0),
        "conf_mid": _make_conformer("conf_mid", tm5_gap=2.0, ionic_gap=3.5, yy_gap=2.0, connector_shift=2.5),
    }
    energies = pd.DataFrame(
        {"structure_id": ["conf_active", "conf_inactive", "conf_mid"], "gibbs_kcal_mol": [-4.0, 0.0, -1.0]}
    )
    return ensemble, energies


@pytest.fixture
def refs():
    active = make_structure("active_ref", "A", {121: {"CA": (0, 0, 0)}, 282: {"CA": (1, 0, 0)}})
    inactive = make_structure("inactive_ref", "A", {121: {"CA": (0, 0, 5)}, 282: {"CA": (1, 0, 5)}})
    return {"active": active, "inactive": inactive}


def test_compute_cv_table_has_expected_columns(ensemble_and_energies, refs):
    ensemble, _ = ensemble_and_energies
    cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
    assert set(cv_table.columns) == {"tm5_bulge", "ionic_lock", "y_y_motif", "connector_drmsd"}
    assert set(cv_table.index) == set(ensemble.keys())
    # conf_active was built exactly matching the active reference -> delta rmsd very negative
    assert cv_table.loc["conf_active", "connector_drmsd"] < 0


def test_merge_with_energies_joins_on_structure_id(ensemble_and_energies, refs):
    ensemble, energies_df = ensemble_and_energies
    cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
    energies = pipeline.merge_with_energies(cv_table, energies_df.set_index("structure_id"))
    assert len(energies) == 3
    assert "gibbs_kcal_mol" in energies.columns


def test_merge_with_energies_no_overlap_raises(ensemble_and_energies, refs):
    ensemble, _ = ensemble_and_energies
    cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
    bad_energies = pd.DataFrame({"structure_id": ["nonexistent"], "gibbs_kcal_mol": [0.0]}).set_index("structure_id")
    with pytest.raises(ValueError):
        pipeline.merge_with_energies(cv_table, bad_energies)


def test_build_1d_and_2d_landscape(ensemble_and_energies, refs):
    ensemble, energies_df = ensemble_and_energies
    cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
    merged = pipeline.merge_with_energies(cv_table, energies_df.set_index("structure_id"))

    landscape_1d = pipeline.build_1d_landscape(merged, "tm5_bulge", method="kde", grid_size=30)
    assert landscape_1d["dG"].shape == (30,)
    assert np.isfinite(landscape_1d["dG"]).all()

    landscape_2d = pipeline.build_2d_landscape(merged, "tm5_bulge", "ionic_lock", method="kde", grid_size=25)
    assert landscape_2d["dG"].shape == (25, 25)


def test_build_embedding_landscape(ensemble_and_energies, refs):
    ensemble, energies_df = ensemble_and_energies
    cv_table = pipeline.compute_cv_table(ensemble, BETA2AR_MICROSWITCHES, refs=refs)
    merged = pipeline.merge_with_energies(cv_table, energies_df.set_index("structure_id"))

    embedding_df, model = pipeline.build_embedding_landscape(
        merged, feature_cols=["tm5_bulge", "ionic_lock", "y_y_motif"], method="pca"
    )
    assert embedding_df.shape == (3, 3)  # 2 PCs + gibbs column
    assert "gibbs_kcal_mol" in embedding_df.columns
