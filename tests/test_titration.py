import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from linkage_pka.titration import (
    GridParams,
    PqrAtom,
    TITRATABLE_RESIDUES,
    build_microstate,
    build_model_compound_atoms,
    charge_delta,
    compute_intrinsic_pka,
    compute_pairwise_coupling,
    compute_solvation_energy,
    load_amber_charges,
    place_titratable_hydrogen,
    read_pqr,
    write_pqr,
)

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"
PDB2PQR_AVAILABLE = shutil.which("pdb2pqr30") is not None
APBS_AVAILABLE = shutil.which("apbs") is not None and PDB2PQR_AVAILABLE


@pytest.fixture(scope="module")
def ci2_pqr(tmp_path_factory):
    if not PDB2PQR_AVAILABLE:
        pytest.skip("requires pdb2pqr30 on PATH")
    work = tmp_path_factory.mktemp("titration_pqr")
    pqr_path = work / "ci2.pqr"
    result = subprocess.run(
        ["pdb2pqr30", "--ff", "AMBER", "--with-ph", "7.0", "--titration-state-method", "propka",
         str(CI2), str(pqr_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return pqr_path


def test_load_amber_charges_has_expected_residue_variants():
    charges = load_amber_charges()
    for rn in ("ASH", "ASP", "GLH", "GLU", "HID", "HIE", "HIP", "LYN", "LYS", "ARG"):
        assert rn in charges
        assert len(charges[rn]) > 0


@pytest.mark.parametrize("resname,neutral_variant,net", [
    ("ASH", None, 0.0), ("ASP", None, -1.0),
    ("GLH", None, 0.0), ("GLU", None, -1.0),
    ("HID", None, 0.0), ("HIE", None, 0.0), ("HIP", None, 1.0),
    ("LYN", None, 0.0), ("LYS", None, 1.0),
])
def test_amber_net_charges_match_expected_protonation_states(resname, neutral_variant, net):
    charges = load_amber_charges()
    total = sum(c for c, _ in charges[resname].values())
    assert total == pytest.approx(net, abs=1e-6)


@pytest.mark.parametrize("resname", list(TITRATABLE_RESIDUES))
def test_charge_delta_is_one_elementary_charge_for_every_titratable_residue(resname):
    charges = load_amber_charges()
    assert charge_delta(resname, charges) == pytest.approx(1.0, abs=1e-6)


def test_read_write_pqr_roundtrip_including_four_char_atom_names():
    # HD11/HD12/HD13 are real 4-character AMBER atom names (Leu/Ile) --
    # this is exactly the case that broke naive fixed-width formatting
    # (atom name butting directly against the resname with no separator).
    atoms = [
        PqrAtom(serial=1, name="N", resname="LEU", resnum=5, x=1.0, y=2.0, z=3.0, charge=-0.4, radius=1.8),
        PqrAtom(serial=2, name="HD11", resname="LEU", resnum=5, x=4.5, y=-6.25, z=7.125, charge=0.1, radius=1.4),
        PqrAtom(serial=3, name="HD12", resname="LEU", resnum=5, x=-1.5, y=2.5, z=-3.5, charge=0.1, radius=1.4),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.pqr"
        write_pqr(path, atoms)
        reread = read_pqr(path)

    assert len(reread) == len(atoms)
    for a, b in zip(atoms, reread):
        assert a.name == b.name
        assert a.resnum == b.resnum
        assert a.charge == pytest.approx(b.charge, abs=1e-4)
        assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z), abs=1e-3)


def test_read_pqr_on_real_pdb2pqr_output_roundtrips(ci2_pqr):
    atoms = read_pqr(ci2_pqr)
    assert len(atoms) > 0
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "rt.pqr"
        write_pqr(path, atoms)
        reread = read_pqr(path)
    assert len(reread) == len(atoms)
    for a, b in zip(atoms, reread):
        assert a.name == b.name and a.resnum == b.resnum
        assert a.charge == pytest.approx(b.charge, abs=1e-4)


def test_build_microstate_deprotonated_asp_matches_base_when_already_deprotonated(ci2_pqr):
    atoms = read_pqr(ci2_pqr)
    amber_charges = load_amber_charges()
    asp_resnums = sorted({a.resnum for a in atoms if a.resname == "ASP"})
    assert asp_resnums, "CI2 must contain at least one ASP for this test"
    resnum = asp_resnums[0]

    base_res = [a for a in atoms if a.resnum == resnum]
    assert not any(a.name == "HD2" for a in base_res)  # PDB2PQR at pH 7 -> deprotonated, no HD2

    deprot = build_microstate(atoms, resnum, "ASP", protonated=False, amber_charges=amber_charges)
    deprot_res = [a for a in deprot if a.resnum == resnum]
    assert len(deprot_res) == len(base_res)  # no atom added/removed
    assert sum(a.charge for a in deprot_res) == pytest.approx(-1.0, abs=1e-6)


def test_build_microstate_protonated_asp_requires_h_position_and_adds_one_atom(ci2_pqr):
    atoms = read_pqr(ci2_pqr)
    amber_charges = load_amber_charges()
    resnum = sorted({a.resnum for a in atoms if a.resname == "ASP"})[0]

    with pytest.raises(ValueError):
        build_microstate(atoms, resnum, "ASP", protonated=True, amber_charges=amber_charges)

    od2 = next(a for a in atoms if a.resnum == resnum and a.name == "OD2")
    h_pos = np.array([od2.x, od2.y, od2.z]) + np.array([0.9, 0.0, 0.0])
    prot = build_microstate(atoms, resnum, "ASP", protonated=True, amber_charges=amber_charges, extra_h_position=h_pos)

    prot_res = [a for a in prot if a.resnum == resnum]
    base_res = [a for a in atoms if a.resnum == resnum]
    assert len(prot_res) == len(base_res) + 1
    assert any(a.name == "HD2" for a in prot_res)
    assert sum(a.charge for a in prot_res) == pytest.approx(0.0, abs=1e-6)
    assert len(prot) == len(atoms) + 1  # rest of the structure untouched in atom count

    hd2 = next(a for a in prot_res if a.name == "HD2")
    assert (hd2.x, hd2.y, hd2.z) == pytest.approx(tuple(h_pos), abs=1e-6)


def test_build_microstate_lys_roundtrips_charge_between_protonated_and_deprotonated(ci2_pqr):
    atoms = read_pqr(ci2_pqr)
    amber_charges = load_amber_charges()
    lys_resnums = sorted({a.resnum for a in atoms if a.resname == "LYS"})
    assert lys_resnums
    resnum = lys_resnums[0]

    base_res = [a for a in atoms if a.resnum == resnum]
    # LYS is PDB2PQR's default (protonated) name -- base should already carry HZ1.
    assert any(a.name == "HZ1" for a in base_res)
    assert sum(a.charge for a in base_res) == pytest.approx(1.0, abs=1e-6)

    deprot = build_microstate(atoms, resnum, "LYS", protonated=False, amber_charges=amber_charges)
    deprot_res = [a for a in deprot if a.resnum == resnum]
    assert not any(a.name == "HZ1" for a in deprot_res)
    assert len(deprot_res) == len(base_res) - 1
    assert sum(a.charge for a in deprot_res) == pytest.approx(0.0, abs=1e-6)

    # and back to protonated (HZ1 already present in the original base atoms, no extra position needed)
    reprot = build_microstate(atoms, resnum, "LYS", protonated=True, amber_charges=amber_charges)
    reprot_res = [a for a in reprot if a.resnum == resnum]
    assert sum(a.charge for a in reprot_res) == pytest.approx(1.0, abs=1e-6)


def test_build_microstate_only_touches_the_target_residue(ci2_pqr):
    atoms = read_pqr(ci2_pqr)
    amber_charges = load_amber_charges()
    resnum = sorted({a.resnum for a in atoms if a.resname == "GLU"})[0]

    deprot = build_microstate(atoms, resnum, "GLU", protonated=False, amber_charges=amber_charges)
    other_before = {(a.resnum, a.name): a.charge for a in atoms if a.resnum != resnum}
    other_after = {(a.resnum, a.name): a.charge for a in deprot if a.resnum != resnum}
    assert other_before == other_after


def test_place_titratable_hydrogen_gives_reasonable_bond_geometry(ci2_pqr):
    atoms = read_pqr(ci2_pqr)
    resnum = sorted({a.resnum for a in atoms if a.resname == "ASP"})[0]
    h_pos = place_titratable_hydrogen(atoms, resnum, "ASP")

    coords = {a.name: np.array([a.x, a.y, a.z]) for a in atoms if a.resnum == resnum}
    d_od2 = np.linalg.norm(h_pos - coords["OD2"])
    d_od1 = np.linalg.norm(h_pos - coords["OD1"])
    d_cg = np.linalg.norm(h_pos - coords["CG"])

    assert d_od2 == pytest.approx(0.96, abs=1e-6)  # exactly the configured O-H bond length
    assert d_od1 > 1.5  # not clashing with the other carboxylate oxygen
    assert d_cg > 1.5   # not clashing with CG


def test_place_titratable_hydrogen_missing_parent_raises():
    # A residue with no OD2 at all (e.g. a stripped-down synthetic fragment).
    atoms = [
        PqrAtom(serial=1, name="CG", resname="ASP", resnum=1, x=0.0, y=0.0, z=0.0, charge=0.8, radius=1.9),
        PqrAtom(serial=2, name="OD1", resname="ASP", resnum=1, x=1.2, y=0.0, z=0.0, charge=-0.8, radius=1.6),
    ]
    with pytest.raises(KeyError):
        place_titratable_hydrogen(atoms, 1, "ASP")


def test_build_model_compound_atoms_includes_real_neighbor_caps(ci2_pqr):
    atoms = read_pqr(ci2_pqr)
    resnum = sorted({a.resnum for a in atoms if a.resname == "ASP"})[0]

    model_atoms = build_model_compound_atoms(atoms, resnum)
    own_atoms = [a for a in model_atoms if a.resnum == resnum]
    prev_cap = [a for a in model_atoms if a.resnum == resnum - 1]
    next_cap = [a for a in model_atoms if a.resnum == resnum + 1]

    assert len(own_atoms) == len([a for a in atoms if a.resnum == resnum])
    assert {a.name for a in prev_cap} <= {"C", "O"}
    assert {a.name for a in next_cap} <= {"N", "H"}
    # the neighbor cap atoms must be the *real* atoms from the structure, not synthesized
    real_prev = {a.name: (a.x, a.y, a.z) for a in atoms if a.resnum == resnum - 1 and a.name in ("C", "O")}
    for a in prev_cap:
        assert (a.x, a.y, a.z) == real_prev[a.name]


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_solvation_energy_is_finite_and_reference_cancels_at_matching_sdie(ci2_pqr, tmp_path):
    atoms = read_pqr(ci2_pqr)
    resnum = sorted({a.resnum for a in atoms if a.resname == "ASP"})[0]
    model_atoms = [a for a in atoms if a.resnum == resnum]

    grid = GridParams(dime=(33, 33, 33), glen=(25.0, 25.0, 25.0), gcent="mol 1",
                       pdie=2.0, sdie=78.54, ion_strength_m=0.150)
    e = compute_solvation_energy(model_atoms, grid, tmp_path / "solv", frame=None)
    assert np.isfinite(e)

    # If sdie == pdie (no dielectric boundary at all), the solvation energy
    # must be ~0 -- solvated and reference calculations become identical.
    grid_no_boundary = GridParams(dime=(33, 33, 33), glen=(25.0, 25.0, 25.0), gcent="mol 1",
                                   pdie=2.0, sdie=2.0, ion_strength_m=0.150)
    e_no_boundary = compute_solvation_energy(model_atoms, grid_no_boundary, tmp_path / "solv2", frame=None)
    assert e_no_boundary == pytest.approx(0.0, abs=1e-3)


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_intrinsic_pka_end_to_end_asp_is_physically_plausible(ci2_pqr, tmp_path):
    """Regression guard for the Born-cycle fix: before it, this calculation
    gave a nonsensical intrinsic pKa around -30 (and got *worse*, not
    better, with finer grids). This asserts a generous sanity envelope
    around the model pKa, not tight numerical accuracy -- that calibration
    is Gate A's job (staphylococcal-nuclease buried-ionizable series), not
    a unit test's."""
    atoms = read_pqr(ci2_pqr)
    resnum = sorted({a.resnum for a in atoms if a.resname == "ASP"})[0]
    h_pos = place_titratable_hydrogen(atoms, resnum, "ASP")

    protein_grid = GridParams(dime=(33, 33, 33), glen=(50.0, 50.0, 50.0), gcent="mol 1",
                               pdie=2.0, sdie=78.54, ion_strength_m=0.150)
    model_grid = GridParams(dime=(33, 33, 33), glen=(25.0, 25.0, 25.0), gcent="mol 1",
                             pdie=2.0, sdie=78.54, ion_strength_m=0.150)

    result = compute_intrinsic_pka(
        atoms, resnum, "ASP", frame=None,
        protein_grid_params=protein_grid, model_grid_params=model_grid,
        work_dir=tmp_path / "pka", extra_h_position=h_pos,
    )
    assert np.isfinite(result.intrinsic_pka)
    assert -5.0 < result.intrinsic_pka < 20.0  # generous sanity envelope, not an accuracy claim
    assert result.model_pka == 3.9


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_pairwise_coupling_runs_and_is_finite(ci2_pqr, tmp_path):
    atoms = read_pqr(ci2_pqr)
    asp_resnums = sorted({a.resnum for a in atoms if a.resname == "ASP"})
    glu_resnums = sorted({a.resnum for a in atoms if a.resname == "GLU"})
    resnum_i, resnum_j = asp_resnums[0], glu_resnums[0]
    h_pos_i = place_titratable_hydrogen(atoms, resnum_i, "ASP")
    h_pos_j = place_titratable_hydrogen(atoms, resnum_j, "GLU")

    grid = GridParams(dime=(33, 33, 33), glen=(50.0, 50.0, 50.0), gcent="mol 1",
                       pdie=2.0, sdie=78.54, ion_strength_m=0.150)

    w_ij = compute_pairwise_coupling(
        atoms, resnum_i, "ASP", resnum_j, "GLU", frame=None, grid_params=grid,
        work_dir=tmp_path / "coupling", extra_h_position_i=h_pos_i, extra_h_position_j=h_pos_j,
    )
    assert np.isfinite(w_ij)
