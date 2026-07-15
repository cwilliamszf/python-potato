import numpy as np

from wsme_gpcr.ionizable_network import _connected_components, compute_ionizable_network
from wsme_gpcr.structure import Structure


def _make_structure(residues):
    """residues: list of (resname, {atom_name: xyz})"""
    resname, author_resnum, atom_name, coord, atom_resindex, charge = [], [], [], [], [], []
    for ridx, (rname, atoms) in enumerate(residues):
        resname.append(rname)
        author_resnum.append(ridx + 1)
        for aname, xyz in atoms.items():
            atom_name.append(aname)
            coord.append(xyz)
            atom_resindex.append(ridx)
            charge.append(0.0)
    return Structure(
        resname=resname,
        seq="A" * len(residues),
        author_resnum=np.array(author_resnum),
        atom_name=atom_name,
        coord=np.array(coord, dtype=float),
        atom_resindex=np.array(atom_resindex),
        charge=np.array(charge),
        bfactor=np.zeros(len(atom_name)),
        chain_id="A",
        ph=7.0,
    )


def test_connected_components_basic():
    # 0-1-2 connected, 3 isolated, 4-5 connected
    edges = [(0, 1), (1, 2), (4, 5)]
    comps = _connected_components(6, edges)
    comp_sets = sorted([sorted(c) for c in comps])
    assert comp_sets == [[0, 1, 2], [3], [4, 5]]


def test_two_close_his_form_a_network_far_one_does_not():
    # Two histidines close together (< 10 A), a third far away (> 10 A);
    # a couple of non-ionizable residues (GLY) that should be ignored.
    residues = [
        ("HIS", {"ND1": [0.0, 0.0, 0.0], "NE2": [1.0, 0.0, 0.5]}),
        ("HIS", {"ND1": [4.0, 0.0, 1.0], "NE2": [5.0, 0.0, 0.0]}),
        ("HIS", {"ND1": [100.0, 0.0, 2.0], "NE2": [101.0, 0.0, 1.0]}),
        ("GLY", {"CA": [2.0, 1.0, 0.5]}),
        ("ASP", {"OD1": [2.0, -3.0, 1.5], "OD2": [2.0, -4.0, 0.5]}),
    ]
    structure = _make_structure(residues)
    result = compute_ionizable_network(structure, edge_cutoff=10.0)

    # Only HIS/ASP residues (3 HIS + 1 ASP) are tracked; GLY is excluded.
    assert len(result.residue_index) == 4
    assert set(result.resname) == {"HIS", "ASP"}

    # The two close HIS + the nearby ASP should form one connected network;
    # the far-away HIS (100+) should be isolated in its own network.
    sizes = sorted(len(n) for n in result.networks)
    assert sizes == [1, 3]


def test_burial_classification_has_three_tiers_when_enough_points():
    rng = np.random.default_rng(0)
    residues = []
    for i in range(12):
        pos = rng.uniform(-20, 20, size=3)
        residues.append(("HIS", {"ND1": pos, "NE2": pos + [1, 0, 0]}))
    structure = _make_structure(residues)
    result = compute_ionizable_network(structure)
    assert len(result.burial_score) == 12
    assert set(result.burial_class) <= {"buried", "margin", "exposed"}
