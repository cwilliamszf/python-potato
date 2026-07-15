import pandas as pd
import pytest
from Bio.PDB import PDBIO

from gpcr_energy_landscapes import io
from tests.helpers import make_structure


def _write_pdb(structure, path):
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))


def test_load_ensemble_keys_by_filename_stem(tmp_path):
    for i in range(3):
        structure = make_structure(f"conf_{i}", "A", {1: {"CA": (float(i), 0, 0)}})
        _write_pdb(structure, tmp_path / f"conf_{i}.pdb")

    ensemble = io.load_ensemble(tmp_path)
    assert set(ensemble.keys()) == {"conf_0", "conf_1", "conf_2"}


def test_load_ensemble_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_ensemble(tmp_path)


def test_load_energies_from_dataframe():
    df = pd.DataFrame({"structure_id": ["a", "b"], "gibbs_kcal_mol": [-1.0, 2.0]})
    energies = io.load_energies(df)
    assert list(energies.index) == ["a", "b"]
    assert energies.loc["a", "gibbs_kcal_mol"] == -1.0


def test_load_energies_missing_column_raises():
    df = pd.DataFrame({"structure_id": ["a"], "not_gibbs": [1.0]})
    with pytest.raises(ValueError):
        io.load_energies(df)


def test_load_energies_from_csv(tmp_path):
    csv_path = tmp_path / "energies.csv"
    csv_path.write_text("structure_id,gibbs_kcal_mol\na,-2.5\nb,0.5\n")
    energies = io.load_energies(csv_path)
    assert energies.loc["b", "gibbs_kcal_mol"] == 0.5
