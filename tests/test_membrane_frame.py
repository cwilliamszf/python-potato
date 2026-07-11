from pathlib import Path

import numpy as np
import pytest

from linkage_pka.membrane_frame import (
    MembraneFrame,
    _looks_like_plddt,
    _segment_tm_helices,
    compute_membrane_frame,
    find_d250,
    find_r350,
    find_y753,
)
from wsme_gpcr.structure import Structure, load_structure

GPR68_INACTIVE = Path("/root/.claude/uploads/e6c23a7d-0f3f-50fe-a92b-cd58fe8f9e63/234c57db-WT_Inactive_GPCRdb.pdb")
CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"


def _make_structure(resnames, bfactor=None):
    """A minimal linear-chain Structure with just CA atoms, for testing
    motif search and PCA geometry without needing a real PDB."""
    n = len(resnames)
    author_resnum = np.arange(1, n + 1)
    atom_name = ["CA"] * n
    # place along the z-axis so the "membrane normal" PCA answer is known
    coord = np.array([[0.0, 0.0, float(i)] for i in range(n)])
    atom_resindex = np.arange(n)
    return Structure(
        resname=resnames,
        seq="A" * n,
        author_resnum=author_resnum,
        atom_name=atom_name,
        coord=coord,
        atom_resindex=atom_resindex,
        charge=np.zeros(n),
        bfactor=np.full(n, 80.0) if bfactor is None else np.full(n, float(bfactor)),
        chain_id="A",
        ph=7.0,
    )


THREE_LETTER = {
    "D": "ASP", "R": "ARG", "Y": "TYR", "N": "ASN", "P": "PRO", "A": "ALA",
    "G": "GLY", "L": "LEU", "E": "GLU",
}


def _resnames_from_seq(seq: str) -> list:
    return [THREE_LETTER[c] for c in seq]


@pytest.mark.skipif(not GPR68_INACTIVE.exists(), reason="requires the uploaded GPR68 structure")
def test_find_r350_and_y753_on_real_gpr68_structure():
    s = load_structure(GPR68_INACTIVE)
    r350, dry = find_r350(s)
    y753, npxxy = find_y753(s)
    assert r350 == 119
    assert dry == "DRY"
    assert y753 == 286
    assert npxxy == "DPVLY"


def test_find_r350_locates_literal_dry_motif():
    seq = "AAAA" + "DRY" + "AAAA"
    s = _make_structure(_resnames_from_seq(seq))
    resnum, motif = find_r350(s)
    assert resnum == 6  # 1-indexed author_resnum of the R in DRY (position 5 0-indexed -> resnum 6)
    assert motif == "DRY"


def test_find_r350_accepts_documented_variant():
    seq = "AAAA" + "ERY" + "AAAA"  # E-R-Y variant
    s = _make_structure(_resnames_from_seq(seq))
    resnum, motif = find_r350(s)
    assert motif == "ERY"


def test_find_r350_raises_when_absent():
    s = _make_structure(_resnames_from_seq("AAAAAAAA"))
    with pytest.raises(ValueError):
        find_r350(s)


def test_find_y753_locates_npxxy_and_dpxxy_variants():
    s_npxxy = _make_structure(_resnames_from_seq("AA" + "NPAAY" + "AA"))
    resnum, motif = find_y753(s_npxxy)
    assert motif == "NPAAY"

    s_dpxxy = _make_structure(_resnames_from_seq("AA" + "DPAAY" + "AA"))
    resnum, motif = find_y753(s_dpxxy)
    assert motif == "DPAAY"


def test_looks_like_plddt_rejects_out_of_range_values():
    assert _looks_like_plddt(np.array([10.0, 50.0, 90.0, 100.0]))
    assert not _looks_like_plddt(np.array([10.0, -5.46, 90.0]))  # matches this repo's real GPR68 B-factor artifact
    assert not _looks_like_plddt(np.array([10.0, 150.0, 90.0]))


@pytest.mark.skipif(not GPR68_INACTIVE.exists(), reason="requires the uploaded GPR68 structure")
def test_compute_membrane_frame_on_real_structure_matches_known_topology():
    """Physical sanity check on the real GPR68 structure: class A GPCRs
    have an extracellular N-terminus and an intracellular DRY motif (end
    of TM3) -- the fitted frame must reproduce that sign convention."""
    s = load_structure(GPR68_INACTIVE)
    frame = compute_membrane_frame(s)

    assert frame.axis.shape == (3,)
    assert np.linalg.norm(frame.axis) == pytest.approx(1.0, abs=1e-8)
    assert frame.tm_mask_method == "secondary_structure_helix"  # this file's B-factor isn't real pLDDT
    assert frame.r350_resnum == 119
    assert frame.y753_resnum == 286

    n_term_ca = s.coord[(s.atom_resindex == 0) & (np.array(s.atom_name) == "CA")][0]
    assert frame.project(n_term_ca) > 0  # extracellular N-terminus

    r350_ridx = int(np.where(s.author_resnum == frame.r350_resnum)[0][0])
    r350_ca = s.coord[(s.atom_resindex == r350_ridx) & (np.array(s.atom_name) == "CA")][0]
    assert frame.project(r350_ca) < 0  # intracellular DRY motif


@pytest.mark.skipif(not GPR68_INACTIVE.exists(), reason="requires the uploaded GPR68 structure")
def test_fit_half_thickness_differs_from_default():
    s = load_structure(GPR68_INACTIVE)
    default_frame = compute_membrane_frame(s, fit_half_thickness=False)
    fitted_frame = compute_membrane_frame(s, fit_half_thickness=True)
    assert default_frame.half_thickness_ang == 15.0
    assert not default_frame.half_thickness_fitted
    assert fitted_frame.half_thickness_fitted
    assert fitted_frame.half_thickness_ang != 15.0
    assert fitted_frame.half_thickness_ang > 0


def test_membrane_frame_project_and_in_slab_synthetic():
    frame = MembraneFrame(
        origin=np.array([0.0, 0.0, 0.0]), axis=np.array([0.0, 0.0, 1.0]),
        half_thickness_ang=10.0, tm_mask_method="plddt", tm_mask_resnums=[],
        plddt_threshold=70.0, r350_resnum=1, dry_motif="DRY", y753_resnum=2,
        npxxy_motif="NPAAY", half_thickness_fitted=False, explained_variance_ratio=1.0,
    )
    points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 15.0], [3.0, 4.0, 0.0]])
    proj = frame.project(points)
    assert proj == pytest.approx([5.0, 15.0, 0.0])
    assert list(frame.in_slab(points)) == [True, False, True]


def test_compute_membrane_frame_too_few_tm_residues_raises():
    # bfactor=-1 is out of pLDDT range, forcing the secondary-structure
    # fallback; this helper's synthetic structures have CA atoms only (no
    # N/C), so phi/psi are all NaN, assign_secondary_structure finds zero
    # helical residues, and the TM mask ends up empty -- must raise rather
    # than silently fit a membrane axis to nothing.
    s = _make_structure(_resnames_from_seq("DRYAAAAAANPAAY"), bfactor=-1.0)
    with pytest.raises(ValueError):
        compute_membrane_frame(s)


# --------------------------------------------- find_d250 / TM segmentation --

GPR68_ACTIVE = Path("/root/.claude/uploads/e6c23a7d-0f3f-50fe-a92b-cd58fe8f9e63/5732e1b4-WT_Active_GPCRdb.pdb")


def _make_structure_z(resnames, z_positions, bfactor=80.0):
    """Like _make_structure, but with explicit per-residue z coordinates
    (x=y=0) instead of the fixed 0,1,2,... spacing -- needed to construct
    precise TM-segment/loop-cap geometry for find_d250 tests."""
    n = len(resnames)
    assert len(z_positions) == n
    return Structure(
        resname=resnames,
        seq="A" * n,
        author_resnum=np.arange(1, n + 1),
        atom_name=["CA"] * n,
        coord=np.array([[0.0, 0.0, float(z)] for z in z_positions]),
        atom_resindex=np.arange(n),
        charge=np.zeros(n),
        bfactor=np.full(n, float(bfactor)),
        chain_id="A",
        ph=7.0,
    )


def _flat_frame(half_thickness_ang, tm_mask_resnums, origin_z=0.0, r350_resnum=1, y753_resnum=2):
    return MembraneFrame(
        origin=np.array([0.0, 0.0, origin_z]), axis=np.array([0.0, 0.0, 1.0]),
        half_thickness_ang=half_thickness_ang, tm_mask_method="plddt", tm_mask_resnums=tm_mask_resnums,
        plddt_threshold=70.0, r350_resnum=r350_resnum, dry_motif="DRY", y753_resnum=y753_resnum,
        npxxy_motif="NPAAY", half_thickness_fitted=False, explained_variance_ratio=1.0,
    )


@pytest.mark.skipif(not GPR68_INACTIVE.exists(), reason="requires the uploaded GPR68 structure")
def test_find_d250_on_real_gpr68_structures():
    for path in (GPR68_INACTIVE, GPR68_ACTIVE):
        if not path.exists():
            continue
        s = load_structure(path)
        frame = compute_membrane_frame(s)
        resnum, candidates = find_d250(s, frame)
        assert resnum == 67
        assert candidates[0][:2] == (67, "ASP")
        # decisive separation from the runner-up on real data, not a near-tie
        assert abs(candidates[0][2]) < 5.0
        assert abs(candidates[1][2]) > 10.0


@pytest.mark.skipif(not GPR68_INACTIVE.exists(), reason="requires the uploaded GPR68 structure")
def test_segment_tm_helices_finds_seven_on_real_structure():
    s = load_structure(GPR68_INACTIVE)
    frame = compute_membrane_frame(s)
    segments = _segment_tm_helices(s, frame)
    assert len(segments) == 7
    tm3_idx = next(i for i, seg in enumerate(segments) if frame.r350_resnum in seg)
    assert tm3_idx == 2  # TM3 is the 3rd segmented helix (0-indexed 2), matching canonical topology


def test_segment_tm_helices_excludes_short_segments():
    n = 60
    resnames = ["ALA"] * n
    z = list(range(n))
    tm_mask = list(range(1, 21)) + list(range(30, 36)) + list(range(40, 61))  # 20, 6, 21 long
    s = _make_structure_z(resnames, z)
    frame = _flat_frame(half_thickness_ang=1000.0, tm_mask_resnums=tm_mask)  # huge slab -> frac filter irrelevant

    segments = _segment_tm_helices(s, frame, min_length=15)
    lengths = sorted(len(seg) for seg in segments)
    assert lengths == [20, 21]  # the 6-residue segment was filtered out


def test_segment_tm_helices_excludes_low_frac_in_slab_segments():
    n = 40
    resnames = ["ALA"] * n
    z = list(range(n))
    tm_mask = list(range(1, 31))  # 30 long, z = 0..29
    s = _make_structure_z(resnames, z)
    # origin far from this segment's z-range -> every projection has |proj| > half_thickness
    frame = _flat_frame(half_thickness_ang=5.0, tm_mask_resnums=tm_mask, origin_z=100.0)

    assert _segment_tm_helices(s, frame, min_length=15) == []


def test_find_d250_picks_candidate_closest_to_membrane_center():
    # TM1 (1-20), TM2 (30-55) with two Asp/Glu candidates, TM3 (60-79) with DRY for R3.50.
    n = 80
    resnames = ["ALA"] * n
    resnames[41] = "GLU"  # resnum 42 (index 41) -- off-center
    resnames[47] = "ASP"  # resnum 48 (index 47) -- near-center
    resnames[59] = "ASP"  # DRY motif start, resnum 60
    resnames[60] = "ARG"
    resnames[61] = "TYR"
    z = list(range(n))
    tm_mask = list(range(1, 21)) + list(range(30, 56)) + list(range(60, 80))
    s = _make_structure_z(resnames, z)
    # origin at z=47 (resnum 48) -> resnum 48 projects to 0 (dead center); resnum 42 projects to -5
    frame = _flat_frame(half_thickness_ang=100.0, tm_mask_resnums=tm_mask, origin_z=47.0, r350_resnum=61)

    resnum, candidates = find_d250(s, frame, r350_resnum=61)
    assert resnum == 48
    assert [c[0] for c in candidates] == [48, 42]  # sorted by |projection|, closest first


def test_find_d250_raises_when_tm2_has_no_asp_glu():
    n = 80
    resnames = ["ALA"] * n
    resnames[59] = "ASP"
    resnames[60] = "ARG"
    resnames[61] = "TYR"
    z = list(range(n))
    tm_mask = list(range(1, 21)) + list(range(30, 56)) + list(range(60, 80))
    s = _make_structure_z(resnames, z)
    frame = _flat_frame(half_thickness_ang=100.0, tm_mask_resnums=tm_mask, origin_z=47.0, r350_resnum=61)

    with pytest.raises(ValueError):
        find_d250(s, frame, r350_resnum=61)


def test_find_d250_raises_when_r350_segment_is_first():
    n = 40
    resnames = ["ALA"] * n
    resnames[0] = "ASP"
    resnames[1] = "ARG"
    resnames[2] = "TYR"
    z = list(range(n))
    tm_mask = list(range(1, 21))  # only one segment, containing the DRY motif
    s = _make_structure_z(resnames, z)
    frame = _flat_frame(half_thickness_ang=1000.0, tm_mask_resnums=tm_mask, r350_resnum=2)

    with pytest.raises(ValueError):
        find_d250(s, frame, r350_resnum=2)


def test_find_d250_raises_when_r350_not_in_any_segment():
    n = 40
    resnames = ["ALA"] * n
    z = list(range(n))
    tm_mask = list(range(1, 21))
    s = _make_structure_z(resnames, z)
    frame = _flat_frame(half_thickness_ang=1000.0, tm_mask_resnums=tm_mask, r350_resnum=35)  # outside the mask

    with pytest.raises(ValueError):
        find_d250(s, frame, r350_resnum=35)


def test_find_d250_warns_on_near_tie_but_still_returns_closest():
    n = 80
    resnames = ["ALA"] * n
    resnames[46] = "GLU"  # resnum 47 -- projects to -1
    resnames[47] = "ASP"  # resnum 48 -- projects to 0 (closest)
    resnames[59] = "ASP"
    resnames[60] = "ARG"
    resnames[61] = "TYR"
    z = list(range(n))
    tm_mask = list(range(1, 21)) + list(range(30, 56)) + list(range(60, 80))
    s = _make_structure_z(resnames, z)
    frame = _flat_frame(half_thickness_ang=100.0, tm_mask_resnums=tm_mask, origin_z=47.0, r350_resnum=61)

    with pytest.warns(UserWarning):
        resnum, candidates = find_d250(s, frame, r350_resnum=61)
    assert resnum == 48  # still picks the closer one despite the near-tie warning
