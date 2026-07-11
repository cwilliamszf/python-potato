from pathlib import Path

import numpy as np
import pytest

from linkage_pka.structure_prep import (
    CHI_ATOMS,
    IONIZABLE_RESNAMES,
    _dihedral_deg,
    _rotate_about_axis,
    _set_chi,
    measure_chi,
    optimize_rotamers,
    run_structure_prep,
)
from linkage_pka.structure_prep import _index_residues, _make_context

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"

# Known ionizable resnums present in CI2.pdb (one per residue type except
# HIS, which CI2 doesn't contain -- HIS chi-atom coverage is exercised by
# test_chi_atom_definitions_are_internally_consistent below instead).
CI2_LYS = 21
CI2_GLU = 23
CI2_ASP = 42
CI2_ARG = 62


def test_dihedral_deg_matches_known_geometry():
    # Four points forming a perfect 90-degree dihedral: p0-p1-p2 in the xy
    # plane, p3 displaced along +z from p2 (standard dihedral convention).
    p0 = np.array([1.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 1.0, 1.0])
    assert _dihedral_deg(p0, p1, p2, p3) == pytest.approx(-90.0, abs=1e-6)


def test_dihedral_deg_zero_for_coplanar_points():
    p0 = np.array([1.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([-1.0, 1.0, 0.0])
    assert abs(_dihedral_deg(p0, p1, p2, p3)) == pytest.approx(180.0, abs=1e-6)


def test_rotate_about_axis_90_degrees():
    # Rotate a point 90 degrees about the z-axis through the origin.
    points = np.array([[1.0, 0.0, 5.0]])
    rotated = _rotate_about_axis(points, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 90.0)
    assert rotated[0] == pytest.approx([0.0, 1.0, 5.0], abs=1e-6)


def test_rotate_about_axis_preserves_distance_to_axis():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5, 3))
    axis_point = np.array([1.0, 2.0, 3.0])
    axis_dir = np.array([1.0, 1.0, 0.0])
    rotated = _rotate_about_axis(points, axis_point, axis_dir, 37.0)
    axis_unit = axis_dir / np.linalg.norm(axis_dir)
    for p, r in zip(points, rotated):
        # Perpendicular distance from the axis line must be preserved by a rotation about it.
        d_before = np.linalg.norm((p - axis_point) - np.dot(p - axis_point, axis_unit) * axis_unit)
        d_after = np.linalg.norm((r - axis_point) - np.dot(r - axis_point, axis_unit) * axis_unit)
        assert d_before == pytest.approx(d_after, abs=1e-6)


def test_chi_atom_definitions_are_internally_consistent():
    """Every chi's rotation axis start (its 2nd defining atom) must never be
    in its own moving set (it's the fixed pivot, not something that moves).
    The 3rd defining atom, by contrast, IS the pivot for the *next* chi and
    so should still move under the *current* chi (e.g. CG moves under chi1,
    since chi1's axis is CA-CB) but not under later chis where it becomes
    the new axis start (e.g. CG must not move under chi2, whose axis is
    CB-CG)."""
    for resname, chis in CHI_ATOMS.items():
        for defining_atoms, moving_names in chis:
            assert len(defining_atoms) == 4
            assert defining_atoms[1] not in moving_names, f"{resname}: axis-start atom must not be in its own moving set"
    # Concrete illustration of the chi1-vs-chi2 distinction above:
    assert "CG" in CHI_ATOMS["ASP"][0][1]       # chi1 (axis CA-CB): CG moves
    assert "CG" not in CHI_ATOMS["ASP"][1][1]   # chi2 (axis CB-CG): CG is the pivot, stays put


def test_measure_chi_and_set_chi_round_trip():
    import pdbfixer
    from openmm import app

    fixer = pdbfixer.PDBFixer(filename=str(CI2))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller.addHydrogens(forcefield, pH=7.0)
    from openmm import unit
    positions_ang = np.asarray(modeller.positions.value_in_unit(unit.angstrom))

    residues = _index_residues(modeller.topology)
    res_index = residues[CI2_LYS]

    for target in (-60.0, 60.0, 179.9):  # avoid exactly +/-180 (atan2 branch edge)
        moved = _set_chi(positions_ang, res_index, 0, target)
        got = measure_chi(moved, res_index, 0)
        assert got == pytest.approx(target, abs=1e-3)

    # Setting chi1 must not move the backbone atoms that define it (N, CA)
    # or the residue's own CB (the rotation axis start).
    moved = _set_chi(positions_ang, res_index, 0, 60.0)
    for name in ("N", "CA", "CB"):
        idx = res_index.name_to_index[name]
        assert np.allclose(moved[idx], positions_ang[idx])


def test_set_chi2_does_not_move_chi1_defining_atoms():
    import pdbfixer
    from openmm import app, unit

    fixer = pdbfixer.PDBFixer(filename=str(CI2))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller.addHydrogens(forcefield, pH=7.0)
    positions_ang = np.asarray(modeller.positions.value_in_unit(unit.angstrom))

    residues = _index_residues(modeller.topology)
    res_index = residues[CI2_GLU]  # GLU has chi1 and chi2

    moved = _set_chi(positions_ang, res_index, 1, 100.0)  # chi2
    for name in ("N", "CA", "CB", "CG"):
        idx = res_index.name_to_index[name]
        assert np.allclose(moved[idx], positions_ang[idx])
    # chi1 value itself should be unchanged by a chi2 edit
    assert measure_chi(moved, res_index, 0) == pytest.approx(measure_chi(positions_ang, res_index, 0), abs=1e-6)


@pytest.mark.parametrize("resnum", [CI2_LYS, CI2_GLU, CI2_ASP, CI2_ARG])
def test_optimize_rotamers_picks_the_scored_minimum(resnum):
    import pdbfixer
    from openmm import app, unit

    fixer = pdbfixer.PDBFixer(filename=str(CI2))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller.addHydrogens(forcefield, pH=7.0)
    positions_ang = np.asarray(modeller.positions.value_in_unit(unit.angstrom))

    system, context = _make_context(modeller.topology, forcefield)
    _, choices = optimize_rotamers(modeller.topology, positions_ang, [resnum], context)

    choice = choices[resnum]
    assert len(choice.chi_chosen) == 2
    for angle in choice.chi_chosen:
        assert any(angle == pytest.approx(s, abs=1e-6) for s in (-60.0, 60.0, 180.0))

    chosen_key = tuple(choice.chi_chosen)
    assert choice.energy_kj_per_mol[chosen_key] == pytest.approx(min(choice.energy_kj_per_mol.values()))
    assert len(choice.energy_kj_per_mol) == 9  # 3x3 staggered chi1 x chi2 grid


def test_optimize_rotamers_unknown_resnum_raises():
    import pdbfixer
    from openmm import app

    fixer = pdbfixer.PDBFixer(filename=str(CI2))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller.addHydrogens(forcefield, pH=7.0)
    from openmm import unit
    positions_ang = np.asarray(modeller.positions.value_in_unit(unit.angstrom))
    system, context = _make_context(modeller.topology, forcefield)

    with pytest.raises(KeyError):
        optimize_rotamers(modeller.topology, positions_ang, [999999], context)


def test_optimize_rotamers_accepts_string_resnum():
    import pdbfixer
    from openmm import app, unit

    fixer = pdbfixer.PDBFixer(filename=str(CI2))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete([a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "H"])
    forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller.addHydrogens(forcefield, pH=7.0)
    positions_ang = np.asarray(modeller.positions.value_in_unit(unit.angstrom))
    system, context = _make_context(modeller.topology, forcefield)

    _, choices = optimize_rotamers(modeller.topology, positions_ang, [str(CI2_LYS)], context)
    assert CI2_LYS in choices  # keyed by int regardless of the input type


def test_run_structure_prep_end_to_end_no_minimization():
    result = run_structure_prep(str(CI2), ionizable_resnums=[CI2_LYS, CI2_ASP], ph=7.0, minimize=False)

    assert set(result.rotamer_choices.keys()) == {CI2_LYS, CI2_ASP}
    assert result.positions_ang.shape[1] == 3
    assert np.array_equal(result.positions_ang, result.positions_ang_pre_minimization)
    # displacement is tracked for every residue regardless of minimize=, but
    # with minimize=False final_positions IS positions_pre_min, so every
    # displacement is exactly zero and nothing is ever "strained".
    assert len(result.ca_displacement_ang) > 0
    assert all(d == pytest.approx(0.0) for d in result.ca_displacement_ang.values())
    assert result.strained_residues == []
    assert result.tool_versions["protonation_ph"] == 7.0
    assert "openmm" in result.tool_versions and "pdbfixer" in result.tool_versions
    assert "NOT the Dunbrack" in result.tool_versions["rotamer_method"]


def test_run_structure_prep_with_minimization_tracks_ca_displacement():
    result = run_structure_prep(
        str(CI2), ionizable_resnums=[CI2_LYS], ph=7.0, minimize=True, ca_tolerance_ang=0.5,
    )
    assert len(result.ca_displacement_ang) > 0
    assert all(d >= 0 for d in result.ca_displacement_ang.values())
    assert not np.array_equal(result.positions_ang, result.positions_ang_pre_minimization)
    for resnum in result.strained_residues:
        assert result.ca_displacement_ang[resnum] > result.ca_tolerance_ang


def test_run_structure_prep_default_scans_every_ionizable_residue():
    result = run_structure_prep(str(CI2), ionizable_resnums=None, ph=7.0, minimize=False)
    resnames = {c.resname for c in result.rotamer_choices.values()}
    assert resnames <= IONIZABLE_RESNAMES
    assert CI2_LYS in result.rotamer_choices
    assert CI2_GLU in result.rotamer_choices
    assert CI2_ASP in result.rotamer_choices
    assert CI2_ARG in result.rotamer_choices
