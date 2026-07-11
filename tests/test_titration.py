import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from linkage_pka.titration import (
    PqrAtom,
    TITRATABLE_RESIDUES,
    build_microstate,
    charge_delta,
    load_amber_charges,
    read_pqr,
    write_pqr,
)

CI2 = Path(__file__).parent.parent / "examples" / "data" / "CI2.pdb"
PDB2PQR_AVAILABLE = shutil.which("pdb2pqr30") is not None


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
