"""Shared test helpers: build minimal synthetic PDB structures in-memory so
tests don't depend on any external/downloaded structural data."""

from __future__ import annotations

import numpy as np
from Bio.PDB.StructureBuilder import StructureBuilder


def make_structure(structure_id: str, chain_id: str, residues: dict) -> "Bio.PDB.Structure.Structure":
    """``residues``: ``{resid: {atom_name: (x, y, z)}}``. All atoms are given
    a plausible element inferred from the atom name's first letter."""
    builder = StructureBuilder()
    builder.init_structure(structure_id)
    builder.init_model(0)
    builder.init_chain(chain_id)
    builder.init_seg(" ")
    for resid, atoms in residues.items():
        builder.init_residue("ALA", " ", resid, " ")
        for name, coord in atoms.items():
            element = "H" if name.startswith("H") else name[0]
            builder.init_atom(name, np.array(coord, dtype=float), 0.0, 1.0, " ", name, element=element)
    return builder.get_structure()
