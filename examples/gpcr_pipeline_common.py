"""Small utilities shared across the per-receptor GPCR pipeline demo scripts.

The receptor-agnostic pipeline logic itself (motif detection, linear
interpolation path building, real Gibbs scoring) lives in
gpcr_pipeline_gpr68_string_demo.py and is reused directly by other
receptors' scripts via import -- nothing GPR68-specific is in those
functions. This module holds the one piece that's genuinely new per
receptor: trimming a structure to a well-ordered residue range before
interpolation, needed when a receptor's two endpoint models disagree
wildly in a disordered terminus (see gpr132's README for why).
"""

from __future__ import annotations

from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder


def trim_structure_to_residue_range(structure, chain_id: str, lo: int, hi: int):
    """Return a new Structure containing only chain ``chain_id`` residues
    with ``lo <= resnum <= hi``, preserving all atom data. Used to exclude
    disordered termini (whose position across two independently-modeled
    endpoint structures is essentially arbitrary, not a real conformational
    signal) before linear interpolation, which would otherwise try to
    interpolate atoms across tens of Angstroms of physically meaningless
    displacement.
    """
    chain = structure[0][chain_id]
    builder = StructureBuilder()
    builder.init_structure(structure.id)
    builder.init_model(0)
    builder.init_chain(chain_id)
    builder.init_seg(" ")
    for res in chain:
        if res.id[0] != " " or not (lo <= res.id[1] <= hi):
            continue
        builder.init_residue(res.get_resname(), res.id[0], res.id[1], res.id[2])
        for atom in res:
            builder.init_atom(
                atom.get_name(), atom.coord, atom.get_bfactor(), atom.get_occupancy(),
                atom.get_altloc(), atom.get_fullname(), element=atom.element,
            )
    return builder.get_structure()


def write_structure(structure, path):
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))
