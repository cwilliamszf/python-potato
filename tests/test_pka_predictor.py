import shutil
from pathlib import Path

import numpy as np
import pytest

from wsme_gpcr.pka_predictor import PropkaNotAvailableError, predict_pka_propka
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.structure import CHARGED_RESIDUES, DEFAULT_PKA, load_structure

HAVE_PROPKA = shutil.which("propka3") is not None
CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"
GPCR9I_PDB = "examples/data/gpcr_landscapes_reference/gpcr9i.pdb"

pytestmark = pytest.mark.skipif(not HAVE_PROPKA, reason="propka3 not installed")


def test_predict_pka_propka_returns_int_keyed_dict_of_charged_residues():
    st = load_structure(CI2)
    overrides = predict_pka_propka(str(CI2))

    assert isinstance(overrides, dict)
    assert len(overrides) > 0
    for resnum, pka in overrides.items():
        assert isinstance(resnum, int)
        assert isinstance(pka, float)
        # every returned resnum must be a real Asp/Glu/His/Lys/Arg in the structure
        matches = [rname for rn, rname in zip(st.author_resnum, st.resname) if rn == resnum]
        assert matches, f"resnum {resnum} not found in structure"
        assert matches[0] in CHARGED_RESIDUES


def test_predict_pka_propka_pkas_differ_from_flat_defaults_for_buried_residues():
    # PROPKA's whole point is to deviate from the flat per-residue-type
    # default for residues with unusual local environments -- on a real
    # folded structure, at least some predictions should differ
    # meaningfully (not just rounding noise) from DEFAULT_PKA.
    overrides = predict_pka_propka(GPCR9I_PDB)
    st = load_structure(GPCR9I_PDB)
    resname_by_num = {int(rn): rname for rn, rname in zip(st.author_resnum, st.resname)}

    deviations = [
        abs(pka - DEFAULT_PKA[resname_by_num[rn]])
        for rn, pka in overrides.items()
        if resname_by_num.get(rn) in DEFAULT_PKA
    ]
    assert deviations, "expected at least one scored charged residue"
    assert max(deviations) > 1.0, "expected at least one residue with a >1 pH-unit environment shift"


def test_predict_pka_propka_accepts_cif_via_temp_conversion(tmp_path):
    # exercise the mmCIF -> temp-PDB conversion path with a minimal
    # synthetic single-residue structure, and confirm no temp file leaks
    cif_text = """\
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ASP A 1 1 ? 0.000 0.000 0.000 1.00 50.00 ? 1 A 1
ATOM 2 C CA . ASP A 1 1 ? 1.458 0.000 0.000 1.00 50.00 ? 1 A 1
ATOM 3 C C . ASP A 1 1 ? 2.009 1.420 0.000 1.00 50.00 ? 1 A 1
ATOM 4 O O . ASP A 1 1 ? 1.251 2.390 0.000 1.00 50.00 ? 1 A 1
ATOM 5 C CB . ASP A 1 1 ? 2.006 -0.766 1.216 1.00 50.00 ? 1 A 1
ATOM 6 C CG . ASP A 1 1 ? 1.454 -2.176 1.226 1.00 50.00 ? 1 A 1
ATOM 7 O OD1 . ASP A 1 1 ? 0.284 -2.397 0.860 1.00 50.00 ? 1 A 1
ATOM 8 O OD2 . ASP A 1 1 ? 2.166 -3.089 1.635 1.00 50.00 ? 1 A 1
"""
    cif_path = tmp_path / "one_asp.cif"
    cif_path.write_text(cif_text)

    before = set(tmp_path.iterdir())
    overrides = predict_pka_propka(str(cif_path))
    after = set(tmp_path.iterdir())

    assert 1 in overrides
    assert after == before  # no leaked temp files in tmp_path itself
    # the underlying tempfile module writes elsewhere; just confirm this
    # call didn't crash and produced a plausible pKa
    assert 0.0 < overrides[1] < 14.0


def test_run_pipeline_use_propka_pka_changes_charge_vs_default():
    r_default = run_pipeline(GPCR9I_PDB, ph=7.0)
    r_propka = run_pipeline(GPCR9I_PDB, ph=7.0, use_propka_pka=True)

    assert not np.allclose(r_default.structure.charge, r_propka.structure.charge)
    # both should still be valid, fully-run pipelines
    assert r_default.block_model.nblocks == r_propka.block_model.nblocks


def test_run_pipeline_explicit_pka_overrides_win_over_propka():
    forced = {1: 9.9}
    r = run_pipeline(GPCR9I_PDB, ph=7.0, use_propka_pka=True, pka_overrides=forced)
    st = r.structure
    ridx = [i for i, rn in enumerate(st.author_resnum) if rn == 1]
    if ridx and st.resname[ridx[0]] in CHARGED_RESIDUES:
        from wsme_gpcr.structure import fraction_charged

        atom_mask = st.atom_resindex == ridx[0]
        expected_frac = fraction_charged(7.0, 9.9, st.resname[ridx[0]])
        total_charge = st.charge[atom_mask].sum()
        assert abs(abs(total_charge) - expected_frac) < 0.05


def test_predict_pka_propka_not_available_raises_not_silent(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "propka.run" or name.startswith("propka"):
            raise ImportError("simulated missing propka")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(PropkaNotAvailableError):
        predict_pka_propka(str(CI2))
