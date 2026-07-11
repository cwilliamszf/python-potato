import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from linkage_pka.titration import (
    COULOMB_CONSTANT_KJ_ANG_PER_MOL_E2,
    R_KJ_PER_MOL_K,
    GridParams,
    PqrAtom,
    TITRATABLE_RESIDUES,
    _pairwise_coulomb_energy,
    _pairwise_repulsion_energy,
    build_microstate,
    build_model_compound_atoms,
    build_na_ion_atom,
    charge_delta,
    compute_cluster_joint_energies,
    compute_environment_energies_ensemble,
    compute_intrinsic_pka,
    compute_pairwise_coupling,
    compute_solvation_energy,
    find_relaxation_neighbors,
    load_amber_charges,
    load_na_ion_parameters,
    optimize_rotamer_for_microstate,
    optimize_rotamers_with_neighbors,
    place_na_ion,
    place_titratable_hydrogen,
    read_pqr,
    select_rotamer_ensemble,
    write_pqr,
)
from linkage_pka.structure_prep import CHI_ATOMS, EXTRA_CHI_ATOMS, _dihedral_deg, _rotate_about_axis

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


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_cluster_joint_energies_matches_pairwise_coupling_double_difference(ci2_pqr, tmp_path):
    """compute_cluster_joint_energies computes the same 4 whole-system
    energies compute_pairwise_coupling does (for a 2-site cluster), just
    returning them directly instead of collapsing them into W_ij -- so its
    own double-difference of the returned energies must reproduce
    compute_pairwise_coupling's W_ij exactly (same physics, different
    packaging)."""
    atoms = read_pqr(ci2_pqr)
    asp_resnums = sorted({a.resnum for a in atoms if a.resname == "ASP"})
    glu_resnums = sorted({a.resnum for a in atoms if a.resname == "GLU"})
    resnum_i, resnum_j = asp_resnums[0], glu_resnums[0]
    h_pos_i = place_titratable_hydrogen(atoms, resnum_i, "ASP")
    h_pos_j = place_titratable_hydrogen(atoms, resnum_j, "GLU")

    grid = GridParams(dime=(33, 33, 33), glen=(50.0, 50.0, 50.0), gcent="mol 1",
                       pdie=2.0, sdie=78.54, ion_strength_m=0.150)

    sites = [(resnum_i, "ASP"), (resnum_j, "GLU")]
    energies = compute_cluster_joint_energies(
        atoms, sites, frame=None, grid_params=grid, work_dir=tmp_path / "joint",
        extra_h_positions={resnum_i: h_pos_i, resnum_j: h_pos_j},
    )
    assert set(energies) == {(False, False), (True, False), (False, True), (True, True)}
    assert all(np.isfinite(e) for e in energies.values())

    w_ij_from_joint = (energies[(True, True)] - energies[(True, False)]
                        - energies[(False, True)] + energies[(False, False)])

    w_ij_direct = compute_pairwise_coupling(
        atoms, resnum_i, "ASP", resnum_j, "GLU", frame=None, grid_params=grid,
        work_dir=tmp_path / "coupling_ref", extra_h_position_i=h_pos_i, extra_h_position_j=h_pos_j,
    )
    assert w_ij_from_joint == pytest.approx(w_ij_direct, rel=1e-6)


# ------------------------------------------------ conformational sampling --

def _synthetic_asp_atoms(resnum=1, include_hd2=False):
    coords = {
        "N": np.array([0.0, 0.0, 0.0]),
        "CA": np.array([1.46, 0.0, 0.0]),
        "CB": np.array([2.0, 1.4, 0.0]),
        "CG": np.array([3.5, 1.4, 0.0]),
        "OD1": np.array([4.0, 2.5, 0.0]),
        "OD2": np.array([4.2, 0.5, 1.0]),
    }
    charges = {"N": -0.52, "CA": 0.04, "CB": -0.02, "CG": 0.62, "OD1": -0.7, "OD2": -0.7}
    atoms = [
        PqrAtom(serial=i + 1, name=name, resname="ASP", resnum=resnum,
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), charge=charges[name], radius=1.7)
        for i, (name, pos) in enumerate(coords.items())
    ]
    if include_hd2:
        h_pos = place_titratable_hydrogen(atoms, resnum, "ASP")
        atoms.append(PqrAtom(serial=100, name="HD2", resname="ASH", resnum=resnum,
                              x=float(h_pos[0]), y=float(h_pos[1]), z=float(h_pos[2]), charge=0.44, radius=0.6))
    return atoms


def _apply_chi_sequence(coords_by_name: dict, resname: str, chi_values) -> dict:
    """Test-only helper mirroring optimize_rotamer_for_microstate's
    sequential rotation exactly (same primitives, same order) -- used only
    to construct scenarios with a known geometric outcome (an engineered
    clash), not to reimplement or duplicate-check the function under
    test's own selection logic."""
    all_chi_atoms = {**CHI_ATOMS, **EXTRA_CHI_ATOMS}
    coords = dict(coords_by_name)
    for k, target_deg in enumerate(chi_values):
        defining_atoms, moving_names = all_chi_atoms[resname][k]
        if not all(n in coords for n in defining_atoms):
            continue
        current = _dihedral_deg(*[coords[n] for n in defining_atoms])
        delta = target_deg - current
        axis_point, axis_end = coords[defining_atoms[1]], coords[defining_atoms[2]]
        moving_names_present = [n for n in moving_names if n in coords]
        if not moving_names_present:
            continue
        moving = np.array([coords[n] for n in moving_names_present])
        rotated = _rotate_about_axis(moving, axis_point, axis_end - axis_point, delta)
        for n, p in zip(moving_names_present, rotated):
            coords[n] = p
    return coords


def test_pairwise_coulomb_energy_matches_manual_formula():
    moving_coords = np.array([[0.0, 0.0, 0.0]])
    moving_charges = np.array([0.5])
    other_coords = np.array([[3.0, 0.0, 0.0]])
    other_charges = np.array([-0.4])
    e = _pairwise_coulomb_energy(moving_coords, moving_charges, other_coords, other_charges, dielectric=2.0)
    expected = COULOMB_CONSTANT_KJ_ANG_PER_MOL_E2 * 0.5 * (-0.4) / (2.0 * 3.0)
    assert e == pytest.approx(expected, rel=1e-10)


def test_pairwise_coulomb_energy_respects_distance_cutoff():
    moving_coords = np.array([[0.0, 0.0, 0.0]])
    moving_charges = np.array([0.5])
    other_coords = np.array([[100.0, 0.0, 0.0]])  # far outside any reasonable cutoff
    other_charges = np.array([-0.4])
    e = _pairwise_coulomb_energy(moving_coords, moving_charges, other_coords, other_charges,
                                  dielectric=2.0, distance_cutoff_ang=15.0)
    assert e == 0.0


def test_pairwise_repulsion_energy_matches_manual_formula():
    moving_coords = np.array([[0.0, 0.0, 0.0]])
    moving_radii = np.array([1.7])
    other_coords = np.array([[3.0, 0.0, 0.0]])
    other_radii = np.array([1.5])
    e = _pairwise_repulsion_energy(moving_coords, moving_radii, other_coords, other_radii, epsilon_kj_mol=1.0)
    sigma = 1.7 + 1.5
    expected = 1.0 * (sigma / 3.0) ** 12
    assert e == pytest.approx(expected, rel=1e-10)


def test_pairwise_repulsion_energy_respects_distance_cutoff():
    moving_coords = np.array([[0.0, 0.0, 0.0]])
    moving_radii = np.array([1.7])
    other_coords = np.array([[100.0, 0.0, 0.0]])
    other_radii = np.array([1.5])
    e = _pairwise_repulsion_energy(moving_coords, moving_radii, other_coords, other_radii,
                                    epsilon_kj_mol=1.0, distance_cutoff_ang=15.0)
    assert e == 0.0


def test_pairwise_repulsion_energy_empty_other_returns_zero():
    moving_coords = np.array([[0.0, 0.0, 0.0]])
    moving_radii = np.array([1.7])
    other_coords = np.array([]).reshape(0, 3)
    other_radii = np.array([])
    assert _pairwise_repulsion_energy(moving_coords, moving_radii, other_coords, other_radii) == 0.0


def test_pairwise_repulsion_energy_dominates_at_short_range():
    # A near-overlap (dist << sigma) must give a huge penalty regardless of
    # how small epsilon is -- this is what lets a steric-only clash (see
    # test_optimize_rotamer_for_microstate_avoids_steric_only_clash) beat
    # out an otherwise-favorable Coulomb score.
    moving_coords = np.array([[0.0, 0.0, 0.0]])
    moving_radii = np.array([1.7])
    other_coords = np.array([[0.5, 0.0, 0.0]])  # well inside the 1.7+1.5=3.2 A contact distance
    other_radii = np.array([1.5])
    e = _pairwise_repulsion_energy(moving_coords, moving_radii, other_coords, other_radii, epsilon_kj_mol=1.0)
    assert e > 1e4  # kJ/mol -- unambiguously dominant over any realistic Coulomb term


def test_optimize_rotamer_for_microstate_avoids_steric_only_clash():
    # A purely steric clash: the "other" atom carries zero charge, so
    # _pairwise_coulomb_energy alone is exactly 0 everywhere and cannot
    # distinguish candidates -- only the repulsion term can detect this,
    # proving it is actually applied (not just present in the signature).
    atoms = _synthetic_asp_atoms()
    coords_by_name = {a.name: np.array([a.x, a.y, a.z]) for a in atoms}

    clash_atoms = []
    for i, chi1 in enumerate((-60.0, 60.0, 180.0)):
        result_coords = _apply_chi_sequence(coords_by_name, "ASP", (chi1, 180.0))
        p = result_coords["OD1"]
        clash_atoms.append(PqrAtom(serial=300 + i, name=f"Y{i}", resname="XXX", resnum=98,
                                    x=float(p[0]), y=float(p[1]), z=float(p[2]), charge=0.0, radius=1.6))

    new_atoms, chosen_chi = optimize_rotamer_for_microstate(
        atoms + clash_atoms, 1, "ASP", dielectric=2.0, distance_cutoff_ang=50.0,
    )
    assert chosen_chi[1] != 180.0


def test_optimize_rotamer_for_microstate_avoids_engineered_clash():
    atoms = _synthetic_asp_atoms()
    coords_by_name = {a.name: np.array([a.x, a.y, a.z]) for a in atoms}

    # Regardless of which chi1 candidate ends up chosen, placing a strong
    # same-sign (repulsive) charge exactly where OD1 would land under
    # chi2=180 (for every one of the 3 chi1 candidates) guarantees chi2=180
    # is the worst choice in the whole 3x3 grid.
    clash_atoms = []
    for i, chi1 in enumerate((-60.0, 60.0, 180.0)):
        result_coords = _apply_chi_sequence(coords_by_name, "ASP", (chi1, 180.0))
        p = result_coords["OD1"]
        clash_atoms.append(PqrAtom(serial=200 + i, name=f"X{i}", resname="XXX", resnum=99,
                                    x=float(p[0]), y=float(p[1]), z=float(p[2]), charge=-5.0, radius=1.0))

    new_atoms, chosen_chi = optimize_rotamer_for_microstate(
        atoms + clash_atoms, 1, "ASP", dielectric=2.0, distance_cutoff_ang=50.0,
    )
    assert chosen_chi[1] != 180.0


def test_optimize_rotamer_for_microstate_preserves_charges_and_other_residues():
    atoms = _synthetic_asp_atoms()
    other = PqrAtom(serial=50, name="CA", resname="GLY", resnum=2, x=10.0, y=10.0, z=10.0, charge=0.1, radius=1.7)
    new_atoms, _ = optimize_rotamer_for_microstate(atoms + [other], 1, "ASP", distance_cutoff_ang=50.0)

    charges_before = {(a.resnum, a.name): a.charge for a in atoms + [other]}
    charges_after = {(a.resnum, a.name): a.charge for a in new_atoms}
    assert charges_before == charges_after

    other_after = next(a for a in new_atoms if a.resnum == 2)
    assert (other_after.x, other_after.y, other_after.z) == (10.0, 10.0, 10.0)  # untouched


def test_optimize_rotamer_for_microstate_updates_titratable_h_geometry():
    # resname is always the canonical type ("ASP"), matching how every
    # caller in this module invokes it -- even for a microstate whose
    # atoms are currently the protonated ASH variant, since CHI_ATOMS and
    # TITRATABLE_RESIDUES are both keyed by the canonical name. HD2 is not
    # in CHI_ATOMS' moving sets, so it must be repositioned via
    # place_titratable_hydrogen after rotation, not left desynced from the
    # rotated OD2.
    atoms = _synthetic_asp_atoms(include_hd2=True)
    new_atoms, chosen_chi = optimize_rotamer_for_microstate(atoms, 1, "ASP", distance_cutoff_ang=50.0)

    coords = {a.name: np.array([a.x, a.y, a.z]) for a in new_atoms}
    d_od2_hd2 = np.linalg.norm(coords["OD2"] - coords["HD2"])
    assert d_od2_hd2 == pytest.approx(0.96, abs=1e-6)  # the configured O-H bond length, not stale


def test_optimize_rotamer_for_microstate_unknown_resname_raises():
    atoms = _synthetic_asp_atoms()
    with pytest.raises(KeyError):
        optimize_rotamer_for_microstate(atoms, 1, "NOTAREALRESNAME")


def test_optimize_rotamer_for_microstate_unknown_resnum_raises():
    atoms = _synthetic_asp_atoms()
    with pytest.raises(KeyError):
        optimize_rotamer_for_microstate(atoms, 999, "ASP")


def test_optimize_rotamer_for_microstate_missing_chi_atom_does_not_crash():
    atoms = [a for a in _synthetic_asp_atoms() if a.name != "OD1"]  # drop a chi2-defining atom
    new_atoms, chosen_chi = optimize_rotamer_for_microstate(atoms, 1, "ASP", distance_cutoff_ang=50.0)
    assert len(new_atoms) == len(atoms)
    assert chosen_chi is not None


# ------------------------------------------------- rotamer ensemble (MCCE-style) --

def test_select_rotamer_ensemble_top_candidate_matches_single_best():
    # ensemble_size=1 must pick exactly the same candidate as
    # optimize_rotamer_for_microstate -- both rank by the same classical
    # proxy, so the top-1 ensemble candidate is that function's answer.
    atoms = _synthetic_asp_atoms()
    best_atoms, best_chi = optimize_rotamer_for_microstate(atoms, 1, "ASP", distance_cutoff_ang=50.0)
    ensemble = select_rotamer_ensemble(atoms, 1, "ASP", ensemble_size=1, distance_cutoff_ang=50.0)
    assert len(ensemble) == 1
    ens_atoms, ens_chi = ensemble[0]
    assert ens_chi == best_chi
    coords_best = {a.name: (a.x, a.y, a.z) for a in best_atoms}
    coords_ens = {a.name: (a.x, a.y, a.z) for a in ens_atoms}
    assert coords_best == coords_ens


def test_select_rotamer_ensemble_returns_requested_size_when_available():
    atoms = _synthetic_asp_atoms()
    ensemble = select_rotamer_ensemble(atoms, 1, "ASP", ensemble_size=4, distance_cutoff_ang=50.0)
    assert len(ensemble) == 4  # ASP has 2 chis x 3 staggered angles = 9 combos, plenty available


def test_select_rotamer_ensemble_caps_at_available_candidate_count():
    atoms = _synthetic_asp_atoms()
    ensemble = select_rotamer_ensemble(atoms, 1, "ASP", ensemble_size=100, distance_cutoff_ang=50.0)
    assert len(ensemble) == 9  # 2 chis x 3 staggered angles, not 100 -- no crash, no padding


def test_select_rotamer_ensemble_preserves_charges_and_other_residues():
    atoms = _synthetic_asp_atoms()
    other = PqrAtom(serial=50, name="CA", resname="GLY", resnum=2, x=10.0, y=10.0, z=10.0, charge=0.1, radius=1.7)
    ensemble = select_rotamer_ensemble(atoms + [other], 1, "ASP", ensemble_size=3, distance_cutoff_ang=50.0)
    charges_before = {(a.resnum, a.name): a.charge for a in atoms + [other]}
    for cand_atoms, _ in ensemble:
        charges_after = {(a.resnum, a.name): a.charge for a in cand_atoms}
        assert charges_before == charges_after
        other_after = next(a for a in cand_atoms if a.resnum == 2)
        assert (other_after.x, other_after.y, other_after.z) == (10.0, 10.0, 10.0)


def test_select_rotamer_ensemble_unknown_resname_raises():
    atoms = _synthetic_asp_atoms()
    with pytest.raises(KeyError):
        select_rotamer_ensemble(atoms, 1, "NOTAREALRESNAME")


def test_compute_environment_energies_ensemble_single_candidate_matches_direct_energy(monkeypatch, tmp_path):
    # ensemble_size=1: G_eff = E_0 - RT*ln(1) = E_0 exactly (log-sum-exp
    # over one term is the identity) -- verified against a monkeypatched
    # compute_solvation_energy so this is a pure math check, no APBS needed.
    from linkage_pka import titration as titration_module

    calls = []

    def fake_solvation_energy(atoms, grid_params, work_dir, frame=None, membrane_dielectric=2.0):
        calls.append(work_dir)
        return -42.0

    monkeypatch.setattr(titration_module, "compute_solvation_energy", fake_solvation_energy)

    atoms = _synthetic_asp_atoms()
    h_pos = place_titratable_hydrogen(atoms, 1, "ASP")
    grid = GridParams(dime=(9, 9, 9), glen=(10.0, 10.0, 10.0), gcent="mol 1",
                       pdie=2.0, sdie=78.54, ion_strength_m=0.150)
    e_deprot, e_prot = compute_environment_energies_ensemble(
        atoms, 1, "ASP", load_amber_charges(), grid, tmp_path / "ens1", extra_h_position=h_pos, ensemble_size=1,
    )
    assert e_deprot == pytest.approx(-42.0, abs=1e-9)
    assert e_prot == pytest.approx(-42.0, abs=1e-9)
    assert len(calls) == 2  # exactly one APBS call per microstate (deprot, prot)


def test_compute_environment_energies_ensemble_lower_than_any_single_candidate(monkeypatch, tmp_path):
    # G_eff = -RT*ln(sum(exp(-E_k/RT))) must be <= min(E_k) for >1 candidate
    # (adding accessible states can only lower the free energy) -- the
    # defining thermodynamic property of the ensemble average vs. picking
    # one candidate.
    from linkage_pka import titration as titration_module

    fake_energies = iter([-40.0, -42.0, -38.0, -41.0] * 2)  # 4 candidates x 2 microstates

    def fake_solvation_energy(atoms, grid_params, work_dir, frame=None, membrane_dielectric=2.0):
        return next(fake_energies)

    monkeypatch.setattr(titration_module, "compute_solvation_energy", fake_solvation_energy)

    atoms = _synthetic_asp_atoms()
    h_pos = place_titratable_hydrogen(atoms, 1, "ASP")
    grid = GridParams(dime=(9, 9, 9), glen=(10.0, 10.0, 10.0), gcent="mol 1",
                       pdie=2.0, sdie=78.54, ion_strength_m=0.150)
    e_deprot, e_prot = compute_environment_energies_ensemble(
        atoms, 1, "ASP", load_amber_charges(), grid, tmp_path / "ens4", extra_h_position=h_pos, ensemble_size=4,
    )
    assert e_deprot < -42.0  # strictly below the best individual candidate (-42.0)
    assert e_prot < -42.0
    assert np.isfinite(e_deprot) and np.isfinite(e_prot)


def test_compute_intrinsic_pka_rejects_ensemble_size_with_optimize_rotamer(tmp_path):
    atoms = _synthetic_asp_atoms()
    grid = GridParams(dime=(9, 9, 9), glen=(10.0, 10.0, 10.0), gcent="mol 1",
                       pdie=2.0, sdie=78.54, ion_strength_m=0.150)
    with pytest.raises(ValueError):
        compute_intrinsic_pka(
            atoms, 1, "ASP", frame=None, protein_grid_params=grid, model_grid_params=grid,
            work_dir=tmp_path, optimize_rotamer=True, ensemble_size=4,
        )


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_intrinsic_pka_with_ensemble_size_runs_and_is_finite(ci2_pqr, tmp_path):
    """End-to-end wiring check: ensemble_size must thread through
    compute_environment_energies_ensemble on both the protein and
    model-compound sides without breaking the calculation (no accuracy
    claim -- this is a plumbing test, not a validated-number test)."""
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
        work_dir=tmp_path / "pka_ensemble", extra_h_position=h_pos, ensemble_size=2,
    )
    assert np.isfinite(result.intrinsic_pka)


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_intrinsic_pka_with_optimize_rotamer_runs_and_is_finite(ci2_pqr, tmp_path):
    """End-to-end wiring check: optimize_rotamer=True must thread through
    compute_environment_energies on both the protein and model-compound
    sides without breaking the calculation (no accuracy claim -- this is
    a plumbing test, not a validated-number test)."""
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
        work_dir=tmp_path / "pka_rotamer", extra_h_position=h_pos, optimize_rotamer=True,
    )
    assert np.isfinite(result.intrinsic_pka)


# ------------------------------------------------- multi-residue relaxation --

def _ca_only_atoms(specs):
    """specs: [(resnum, resname, (x,y,z)), ...] -- CA-only stand-ins,
    sufficient for find_relaxation_neighbors (which only reads CA atoms)."""
    return [
        PqrAtom(serial=i + 1, name="CA", resname=resname, resnum=resnum,
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), charge=0.0, radius=1.7)
        for i, (resnum, resname, pos) in enumerate(specs)
    ]


def _synthetic_leu_atoms(resnum, center=(0.0, 0.0, 0.0)):
    c = np.array(center)
    coords = {
        "N": c + np.array([0.0, 0.0, 0.0]),
        "CA": c + np.array([1.46, 0.0, 0.0]),
        "CB": c + np.array([2.0, 1.4, 0.0]),
        "CG": c + np.array([3.5, 1.4, 0.0]),
        "CD1": c + np.array([4.0, 2.5, 0.0]),
        "CD2": c + np.array([4.2, 0.5, 1.0]),
    }
    return [
        PqrAtom(serial=i + 1, name=name, resname="LEU", resnum=resnum,
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), charge=0.0, radius=1.7)
        for i, (name, pos) in enumerate(coords.items())
    ]


def test_find_relaxation_neighbors_filters_by_radius():
    atoms = _ca_only_atoms([
        (1, "ASP", (0, 0, 0)),
        (2, "LEU", (5, 0, 0)),   # within default 8 A radius
        (3, "SER", (20, 0, 0)),  # far outside
    ])
    neighbors = find_relaxation_neighbors(atoms, 1)
    assert neighbors == [(2, "LEU")]


def test_find_relaxation_neighbors_excludes_residues_without_chi_geometry():
    atoms = _ca_only_atoms([
        (1, "ASP", (0, 0, 0)),
        (2, "GLY", (3, 0, 0)),  # close, but no rotatable side chain at all
    ])
    assert find_relaxation_neighbors(atoms, 1) == []


def test_find_relaxation_neighbors_canonicalizes_protonation_variants():
    atoms = _ca_only_atoms([
        (1, "ASP", (0, 0, 0)),
        (2, "HIE", (4, 0, 0)),  # a His protonation-state variant, not "HIS" literally
    ])
    assert find_relaxation_neighbors(atoms, 1) == [(2, "HIS")]


def test_find_relaxation_neighbors_sorted_by_distance():
    atoms = _ca_only_atoms([
        (1, "ASP", (0, 0, 0)),
        (2, "LEU", (6, 0, 0)),
        (3, "SER", (3, 0, 0)),
    ])
    assert find_relaxation_neighbors(atoms, 1) == [(3, "SER"), (2, "LEU")]


def test_find_relaxation_neighbors_missing_target_ca_raises():
    atoms = _ca_only_atoms([(2, "LEU", (5, 0, 0))])
    with pytest.raises(KeyError):
        find_relaxation_neighbors(atoms, 1)


def test_optimize_rotamers_with_neighbors_relaxes_target_and_neighbor():
    target = _synthetic_asp_atoms(resnum=1)
    # Position LEU's CA close to ASP's CA (well within an 8 A neighbor radius).
    neighbor = _synthetic_leu_atoms(resnum=2, center=(3.0, -3.0, 0.0))
    atoms = target + neighbor

    new_atoms, chi_choices = optimize_rotamers_with_neighbors(atoms, 1, "ASP", neighbor_radius_ang=8.0)

    assert set(chi_choices) == {1, 2}  # both target and the one real neighbor got relaxed
    assert len(new_atoms) == len(atoms)


def test_optimize_rotamers_with_neighbors_avoids_clash_on_the_neighbor_itself():
    # The engineered clash sits on LEU's own moving atoms, not on the ASP
    # target -- proving the neighbor's rotamer is actually being
    # independently optimized, not just carried along unchanged.
    target = _synthetic_asp_atoms(resnum=1)
    neighbor = _synthetic_leu_atoms(resnum=2, center=(3.0, -3.0, 0.0))
    leu_coords_by_name = {a.name: np.array([a.x, a.y, a.z]) for a in neighbor}

    clash_atoms = []
    for i, chi1 in enumerate((-60.0, 60.0, 180.0)):
        result_coords = _apply_chi_sequence(leu_coords_by_name, "LEU", (chi1, 180.0))
        p = result_coords["CD1"]
        clash_atoms.append(PqrAtom(serial=400 + i, name=f"Z{i}", resname="XXX", resnum=99,
                                    x=float(p[0]), y=float(p[1]), z=float(p[2]), charge=0.0, radius=1.6))

    atoms = target + neighbor + clash_atoms
    new_atoms, chi_choices = optimize_rotamers_with_neighbors(atoms, 1, "ASP", neighbor_radius_ang=8.0)

    assert chi_choices[2][1] != 180.0  # LEU (resnum 2)'s chi2 avoided the engineered clash


def test_optimize_rotamers_with_neighbors_leaves_distant_residues_untouched():
    target = _synthetic_asp_atoms(resnum=1)
    neighbor = _synthetic_leu_atoms(resnum=2, center=(3.0, -3.0, 0.0))
    distant = PqrAtom(serial=99, name="CA", resname="SER", resnum=50, x=500.0, y=500.0, z=500.0,
                       charge=0.0, radius=1.7)
    atoms = target + neighbor + [distant]

    new_atoms, chi_choices = optimize_rotamers_with_neighbors(atoms, 1, "ASP", neighbor_radius_ang=8.0)

    assert 50 not in chi_choices
    distant_after = next(a for a in new_atoms if a.resnum == 50)
    assert (distant_after.x, distant_after.y, distant_after.z) == (500.0, 500.0, 500.0)


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_compute_intrinsic_pka_with_neighbor_relaxation_runs_and_is_finite(ci2_pqr, tmp_path):
    """End-to-end wiring check for neighbor_radius_ang -- plumbing only,
    not an accuracy claim."""
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
        work_dir=tmp_path / "pka_neighbor_rotamer", extra_h_position=h_pos,
        optimize_rotamer=True, neighbor_radius_ang=8.0,
    )
    assert np.isfinite(result.intrinsic_pka)


# --------------------------------------------------------- Na+ ion modeling --

def _synthetic_glu_atoms(resnum=1):
    coords = {
        "N": np.array([0.0, 0.0, 0.0]),
        "CA": np.array([1.46, 0.0, 0.0]),
        "CB": np.array([2.0, 1.4, 0.0]),
        "CG": np.array([3.0, 1.6, 0.5]),
        "CD": np.array([4.0, 2.5, 0.0]),
        "OE1": np.array([4.5, 3.5, 0.0]),
        "OE2": np.array([4.3, 1.5, 1.2]),
    }
    charges = {"N": -0.52, "CA": 0.04, "CB": -0.02, "CG": 0.01, "CD": 0.62, "OE1": -0.7, "OE2": -0.7}
    return [
        PqrAtom(serial=i + 1, name=name, resname="GLU", resnum=resnum,
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), charge=charges[name], radius=1.7)
        for i, (name, pos) in enumerate(coords.items())
    ]


def test_load_na_ion_parameters_are_physically_reasonable():
    charge, radius = load_na_ion_parameters()
    assert charge == pytest.approx(1.0, abs=1e-9)  # Na+ is a monovalent cation, exactly
    assert 0.5 < radius < 3.0  # a real ionic radius, not a placeholder/degenerate value


def test_place_na_ion_roughly_equidistant_from_both_oxygens():
    # A chemically symmetric carboxylate (OD1/OD2 truly equidistant from
    # CG) -- _synthetic_asp_atoms' geometry is asymmetric (built for
    # rotamer tests, where that didn't matter), so it isn't a valid
    # fixture for a bisector-symmetry claim.
    atoms = [
        PqrAtom(serial=1, name="CG", resname="ASP", resnum=1, x=0.0, y=0.0, z=0.0, charge=0.62, radius=1.9),
        PqrAtom(serial=2, name="OD1", resname="ASP", resnum=1, x=1.0, y=0.75, z=0.0, charge=-0.7, radius=1.6612),
        PqrAtom(serial=3, name="OD2", resname="ASP", resnum=1, x=1.0, y=-0.75, z=0.0, charge=-0.7, radius=1.6612),
    ]
    pos = place_na_ion(atoms, 1, "ASP")
    od1 = next(a for a in atoms if a.name == "OD1")
    od2 = next(a for a in atoms if a.name == "OD2")
    d1 = np.linalg.norm(pos - np.array([od1.x, od1.y, od1.z]))
    d2 = np.linalg.norm(pos - np.array([od2.x, od2.y, od2.z]))
    assert d1 == pytest.approx(d2, rel=1e-6)  # placed exactly on the bisector by construction


def test_place_na_ion_uses_lj_combining_rule_contact_distance():
    atoms = _synthetic_asp_atoms()
    pos = place_na_ion(atoms, 1, "ASP")
    od1 = next(a for a in atoms if a.name == "OD1")
    od2 = next(a for a in atoms if a.name == "OD2")
    midpoint = np.array([(od1.x + od2.x) / 2, (od1.y + od2.y) / 2, (od1.z + od2.z) / 2])

    _, na_radius = load_na_ion_parameters()
    expected_dist_from_midpoint = na_radius + od1.radius  # od1/od2 share the same AMBER radius
    assert np.linalg.norm(pos - midpoint) == pytest.approx(expected_dist_from_midpoint, rel=1e-6)


def test_place_na_ion_works_for_glu():
    atoms = _synthetic_glu_atoms()
    pos = place_na_ion(atoms, 1, "GLU")
    assert np.all(np.isfinite(pos))


def test_place_na_ion_missing_atoms_raises():
    atoms = [a for a in _synthetic_asp_atoms() if a.name != "OD1"]
    with pytest.raises(KeyError):
        place_na_ion(atoms, 1, "ASP")


def test_build_na_ion_atom_has_sentinel_identity_and_real_charge():
    atoms = _synthetic_asp_atoms()
    ion = build_na_ion_atom(atoms, 1, "ASP")
    assert ion.resname == "NA"
    assert ion.resnum == -1  # never collides with a real 1-indexed protein resnum
    assert ion.charge == pytest.approx(1.0, abs=1e-9)
    _, expected_radius = load_na_ion_parameters()
    assert ion.radius == pytest.approx(expected_radius)


def test_build_na_ion_atom_excluded_from_titratable_site_identification():
    from linkage_pka.pipeline import identify_titratable_sites
    atoms = _synthetic_asp_atoms()
    ion = build_na_ion_atom(atoms, 1, "ASP")
    sites = identify_titratable_sites(atoms + [ion])
    assert (-1, "NA") not in sites
    assert all(resname != "NA" for _, resname in sites)


@pytest.mark.skipif(not APBS_AVAILABLE, reason="requires apbs and pdb2pqr30 on PATH")
def test_na_ion_changes_pka_with_vs_without(ci2_pqr, tmp_path):
    """The actual with/without comparison the pipeline spec asks for:
    running the same site's intrinsic pKa calculation with and without
    the modeled ion appended to protein_atoms must give two finite,
    genuinely different results -- proving the ion is actually being
    picked up by the PB solve, not silently ignored."""
    atoms = read_pqr(ci2_pqr)
    resnum = sorted({a.resnum for a in atoms if a.resname == "ASP"})[0]
    h_pos = place_titratable_hydrogen(atoms, resnum, "ASP")
    ion = build_na_ion_atom(atoms, resnum, "ASP")

    protein_grid = GridParams(dime=(33, 33, 33), glen=(50.0, 50.0, 50.0), gcent="mol 1",
                               pdie=2.0, sdie=78.54, ion_strength_m=0.150)
    model_grid = GridParams(dime=(33, 33, 33), glen=(25.0, 25.0, 25.0), gcent="mol 1",
                             pdie=2.0, sdie=78.54, ion_strength_m=0.150)

    without_ion = compute_intrinsic_pka(
        atoms, resnum, "ASP", frame=None,
        protein_grid_params=protein_grid, model_grid_params=model_grid,
        work_dir=tmp_path / "without_ion", extra_h_position=h_pos,
    )
    with_ion = compute_intrinsic_pka(
        atoms + [ion], resnum, "ASP", frame=None,
        protein_grid_params=protein_grid, model_grid_params=model_grid,
        work_dir=tmp_path / "with_ion", extra_h_position=h_pos,
    )
    assert np.isfinite(without_ion.intrinsic_pka)
    assert np.isfinite(with_ion.intrinsic_pka)
    assert without_ion.intrinsic_pka != pytest.approx(with_ion.intrinsic_pka, abs=1e-6)
