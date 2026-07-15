from pathlib import Path

import numpy as np
import pytest

from wsme_gpcr.ion_pocket import add_ion_pocket_interaction, place_na_ion, place_na_ion_multi_coordinate
from wsme_gpcr.pipeline import run_pipeline
from wsme_gpcr.structure import Structure

INACTIVE_PDB = "/root/.claude/uploads/e6c23a7d-0f3f-50fe-a92b-cd58fe8f9e63/e7990488-WT_Inactive_GPCRdb.pdb"
HAVE_GPR68 = Path(INACTIVE_PDB).exists()


def _synthetic_asp_structure(resnum=67, other_resnum=200, other_dist_ang=4.0, other_charge=-0.5):
    # A minimal ASP (N, CA, CB, CG, OD1, OD2) at a fixed, known geometry,
    # plus one other charged atom placed at a controlled distance from
    # where the ion will land, so the Coulomb math can be hand-checked.
    coords = {
        "N": [0.0, 0.0, 0.0], "CA": [1.46, 0.0, 0.0], "CB": [2.0, 1.4, 0.0],
        "CG": [3.5, 1.4, 0.0], "OD1": [4.0, 2.5, 0.0], "OD2": [4.2, 0.5, 1.0],
    }
    resname = ["ASP"]
    author_resnum = [resnum]
    atom_name, coord, atom_resindex, charge = [], [], [], []
    for i, (name, pos) in enumerate(coords.items()):
        atom_name.append(name)
        coord.append(pos)
        atom_resindex.append(0)
        charge.append(-0.5 if name in ("OD1", "OD2") else 0.0)

    # Place the "other" residue's single charged atom at a known distance
    # from the ion position by first computing the ion position from the
    # ASP geometry above, then offsetting along an arbitrary direction.
    partial = Structure(resname=resname, seq="D", author_resnum=np.array(author_resnum),
                         atom_name=atom_name, coord=np.array(coord), atom_resindex=np.array(atom_resindex),
                         charge=np.array(charge), bfactor=np.zeros(len(atom_name)), chain_id="A", ph=7.0)
    ion_pos = place_na_ion(partial, resnum, "ASP")

    other_pos = ion_pos + np.array([other_dist_ang, 0.0, 0.0])
    resname.append("GLU")
    author_resnum.append(other_resnum)
    atom_name.append("OE1")
    coord.append(other_pos.tolist())
    atom_resindex.append(1)
    charge.append(other_charge)

    return Structure(resname=resname, seq="DE", author_resnum=np.array(author_resnum),
                      atom_name=atom_name, coord=np.array(coord), atom_resindex=np.array(atom_resindex),
                      charge=np.array(charge), bfactor=np.zeros(len(atom_name)), chain_id="A", ph=7.0), ion_pos


def _synthetic_block_model(nres, block_of_residue, block_elec=None):
    from wsme_gpcr.blocking import BlockModel
    nblocks = int(block_of_residue.max()) + 1
    return BlockModel(
        nres=nres, nblocks=nblocks, block_size=4, block_of_residue=block_of_residue,
        block_residue_range=np.zeros((nblocks, 2), dtype=int),
        block_cmap=np.zeros((nblocks, nblocks)),
        block_elec=block_elec if block_elec is not None else np.zeros((0, 5)),
    )


def test_place_na_ion_matches_linkage_pka_geometry_convention():
    structure, ion_pos = _synthetic_asp_structure()
    od1 = structure.coord[structure.atom_name.index("OD1")]
    od2 = structure.coord[structure.atom_name.index("OD2")]
    cg = structure.coord[structure.atom_name.index("CG")]
    midpoint = (od1 + od2) / 2.0
    # Ion must sit on the ray from CG through the oxygen midpoint, beyond it.
    to_ion = ion_pos - cg
    to_mid = midpoint - cg
    cos_angle = np.dot(to_ion, to_mid) / (np.linalg.norm(to_ion) * np.linalg.norm(to_mid))
    assert cos_angle == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(ion_pos - cg) > np.linalg.norm(to_mid)
    # Real LJ contact distance: 1.369 (Na+) + 1.6612 (ASP O) Angstrom.
    assert np.linalg.norm(ion_pos - midpoint) == pytest.approx(1.369 + 1.6612, abs=1e-6)


def test_add_ion_pocket_interaction_finds_the_planted_partner():
    # multi_coordinate=False: the naive single-residue ion position, so
    # the planted partner's distance/energy are exactly hand-checkable.
    structure, ion_pos = _synthetic_asp_structure(other_dist_ang=4.0, other_charge=-0.5)
    block_of_residue = np.array([0, 1])  # ASP in block 0, GLU in block 1
    block_model = _synthetic_block_model(nres=2, block_of_residue=block_of_residue)

    result = add_ion_pocket_interaction(structure, block_model, d250_author_resnum=67,
                                         interaction_cutoff_ang=6.0, multi_coordinate=False)
    assert len(result.partners) == 1
    partner = result.partners[0]
    assert partner.author_resnum == 200
    assert partner.dist_to_ion_ang == pytest.approx(4.0, abs=1e-6)
    # Real vacuum Coulomb: COULOMB_CONST * (+1)*(-0.5) / 4.0
    expected_energy = (332.0 * 4.184 / 4.0) * 1.0 * (-0.5) / 4.0
    assert partner.ion_atom_energy_kj_mol == pytest.approx(expected_energy, rel=1e-9)
    assert expected_energy < 0  # attractive: opposite-sign ion/anion


def test_add_ion_pocket_interaction_respects_cutoff():
    structure, ion_pos = _synthetic_asp_structure(other_dist_ang=10.0)
    block_of_residue = np.array([0, 1])
    block_model = _synthetic_block_model(nres=2, block_of_residue=block_of_residue)
    result = add_ion_pocket_interaction(structure, block_model, d250_author_resnum=67,
                                         interaction_cutoff_ang=6.0, multi_coordinate=False)
    assert len(result.partners) == 0
    assert result.new_block_elec_rows.shape == (0, 5)


def test_multi_coordinate_placement_degrades_to_naive_when_no_partner_nearby():
    structure, _ = _synthetic_asp_structure(other_dist_ang=50.0)
    naive_pos = place_na_ion(structure, 67, "ASP")
    refined_pos, coordinating = place_na_ion_multi_coordinate(structure, 67, "ASP")
    np.testing.assert_array_almost_equal(refined_pos, naive_pos)
    assert coordinating == []


def test_multi_coordinate_placement_is_the_centroid_of_the_real_coordinating_oxygens():
    # The refined position must be the centroid of D2.50's own OD1/OD2
    # PLUS the partner oxygen -- not a two-point average of the naive
    # estimate and the partner (that was tried and rejected: see the next
    # test for why it doesn't actually fix a too-close naive point).
    structure, _ = _synthetic_asp_structure(other_dist_ang=4.0, other_charge=-0.5)
    naive_pos = place_na_ion(structure, 67, "ASP")
    refined_pos, coordinating = place_na_ion_multi_coordinate(structure, 67, "ASP")
    assert len(coordinating) == 1
    partner_pos = structure.coord[coordinating[0]]
    od1, od2 = np.array([4.0, 2.5, 0.0]), np.array([4.2, 0.5, 1.0])  # from _synthetic_asp_structure's geometry
    expected = (od1 + od2 + partner_pos) / 3.0
    np.testing.assert_array_almost_equal(refined_pos, expected)
    assert not np.allclose(refined_pos, naive_pos)


def test_multi_coordinate_placement_fixes_the_real_gpr68_unphysical_distance():
    # Regression test for the real finding: the naive placement put the
    # ion 1.33 A from Asp282's OD2 -- shorter than a real covalent bond,
    # impossible for two non-bonded heavy atoms. The refined centroid
    # (of the real coordinating oxygens, not a two-point average with the
    # naive point) must land in a physically plausible Na-O range.
    structure, _ = _synthetic_asp_structure(other_dist_ang=1.33, other_charge=-0.5)
    refined_pos, coordinating = place_na_ion_multi_coordinate(structure, 67, "ASP")
    assert len(coordinating) == 1
    partner_pos = structure.coord[coordinating[0]]
    refined_dist = np.linalg.norm(refined_pos - partner_pos)
    assert refined_dist == pytest.approx(2.756, abs=0.01)
    assert 2.0 <= refined_dist <= 3.0  # a physically plausible Na-O coordination range


def test_zero_partners_leaves_block_elec_unchanged_bit_for_bit():
    # The critical control: no qualifying partner found -> the augmented
    # block_model's block_elec must be IDENTICAL to the original.
    structure, ion_pos = _synthetic_asp_structure(other_dist_ang=50.0)
    block_of_residue = np.array([0, 1])
    original_elec = np.array([[0.0, 1.0, 5.0, 1.0, -2.0]])  # a pre-existing unrelated pair
    block_model = _synthetic_block_model(nres=2, block_of_residue=block_of_residue, block_elec=original_elec)

    result = add_ion_pocket_interaction(structure, block_model, d250_author_resnum=67,
                                         interaction_cutoff_ang=6.0)
    assert result.new_block_elec_rows.shape == (0, 5)
    np.testing.assert_array_equal(result.block_model.block_elec, original_elec)
    # Original block_model object itself must be untouched (a copy, not a mutation).
    np.testing.assert_array_equal(block_model.block_elec, original_elec)


def test_added_rows_are_appended_not_replacing_existing_pairs():
    structure, ion_pos = _synthetic_asp_structure(other_dist_ang=4.0)
    block_of_residue = np.array([0, 1])
    original_elec = np.array([[0.0, 1.0, 5.0, 1.0, -2.0]])
    block_model = _synthetic_block_model(nres=2, block_of_residue=block_of_residue, block_elec=original_elec)

    result = add_ion_pocket_interaction(structure, block_model, d250_author_resnum=67,
                                         interaction_cutoff_ang=6.0)
    assert result.block_model.block_elec.shape[0] == 2  # original + 1 new
    np.testing.assert_array_equal(result.block_model.block_elec[0], original_elec[0])


def test_unknown_resnum_raises_keyerror():
    structure, _ = _synthetic_asp_structure()
    block_model = _synthetic_block_model(nres=2, block_of_residue=np.array([0, 1]))
    with pytest.raises(KeyError):
        add_ion_pocket_interaction(structure, block_model, d250_author_resnum=999)


@pytest.mark.skipif(not HAVE_GPR68, reason="requires the uploaded GPR68 inactive structure")
def test_real_gpr68_finds_d250_asp67_and_at_least_one_partner():
    """Plumbing test on the real structure: does the pipeline find Asp67
    and locate a real ion position with real nearby charged partners
    (no claim about which specific residues -- that's data-driven, see
    the actual investigation script for the reported partner list)."""
    r = run_pipeline(INACTIVE_PDB, ph=7.0)
    result = add_ion_pocket_interaction(r.structure, r.block_model, d250_author_resnum=67,
                                         interaction_cutoff_ang=6.0)
    assert result.ion_position.shape == (3,)
    assert np.all(np.isfinite(result.ion_position))
    assert result.block_model.block_elec.shape[0] >= r.block_model.block_elec.shape[0]
