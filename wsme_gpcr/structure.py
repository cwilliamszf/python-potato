"""PDB/mmCIF structure loading and charge assignment.

Ports the atom-selection and charge-assignment logic in
``cmapCalcElecBlock.m`` (see AthiNaganathan/WSMEmodel and
AthiNaganathan/GPCR-Landscapes) to Python, using Biopython for robust
structure parsing instead of fixed-column text parsing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STANDARD_RESIDUES = {
    "GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TYR", "TRP", "SER",
    "ASP", "ASN", "THR", "GLU", "GLN", "HIS", "LYS", "ARG", "PRO", "CYS",
}

RESNAME_TO_CODE = {
    "GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I", "MET": "M",
    "PHE": "F", "TYR": "Y", "TRP": "W", "SER": "S", "ASP": "D", "ASN": "N",
    "THR": "T", "GLU": "E", "GLN": "Q", "HIS": "H", "LYS": "K", "ARG": "R",
    "PRO": "P", "CYS": "C",
}

# Residues that carry a titratable side-chain charge, and the specific
# atoms across which the unit charge is distributed. Matches charres /
# atomc / charmag{7,5,3.5,2} in cmapCalcElecBlock.m.
CHARGED_RESIDUES = {"HIS", "LYS", "ARG", "GLU", "ASP"}
CHARGE_ATOMS = ["NE", "NH1", "NH2", "NZ", "OD1", "OD2", "OE1", "OE2", "ND1", "NE2"]
CHARGE_TABLE = {
    7.0: [0.33, 0.33, 0.33, 1.0, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0],
    5.0: [0.33, 0.33, 0.33, 1.0, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5],
    3.5: [0.33, 0.33, 0.33, 1.0, -0.25, -0.25, -0.25, -0.25, 0.5, 0.5],
    2.0: [0.33, 0.33, 0.33, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
}


@dataclass
class Structure:
    """Curated heavy-atom structure for one chain of one model."""

    resname: list  # length nres, 3-letter codes
    seq: str  # length nres, 1-letter sequence
    author_resnum: np.ndarray  # (nres,) original PDB residue numbers
    atom_name: list  # (natoms,)
    coord: np.ndarray  # (natoms, 3)
    atom_resindex: np.ndarray  # (natoms,) 0-based residue index per atom
    charge: np.ndarray  # (natoms,) charge magnitude per atom (0 if none)
    chain_id: str
    ph: float
    gaps: list = field(default_factory=list)  # author numbering gaps found

    @property
    def nres(self) -> int:
        return len(self.resname)


def _pick_chain(structure_model, chain_id):
    chains = list(structure_model)
    if chain_id is not None:
        for ch in chains:
            if ch.id == chain_id:
                return ch
        raise ValueError(f"Chain '{chain_id}' not found; available: {[c.id for c in chains]}")
    # Default to the first chain containing standard amino acids.
    for ch in chains:
        if any(res.get_resname() in STANDARD_RESIDUES for res in ch):
            return ch
    raise ValueError("No chain with standard amino acid residues found")


def load_structure(path, chain: str | None = None, model: int = 0, ph: float = 7.0) -> Structure:
    """Load a PDB or mmCIF file into a curated heavy-atom Structure.

    Only ATOM records for the 20 standard amino acids are kept; hydrogens,
    waters, ligands, and alternate (non-primary) conformers are dropped.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    if ph not in CHARGE_TABLE:
        raise ValueError(f"ph must be one of {sorted(CHARGE_TABLE)}, got {ph}")
    charmag = CHARGE_TABLE[ph]

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if path.suffix.lower() in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        bio_structure = parser.get_structure(path.stem, str(path))

    bio_model = bio_structure[model]
    bio_chain = _pick_chain(bio_model, chain)

    resname, author_resnum = [], []
    atom_name, coord, atom_resindex, charge = [], [], [], []

    ridx = 0
    for residue in bio_chain:
        hetflag, _, _ = residue.get_id()
        rname = residue.get_resname()
        if hetflag != " " or rname not in STANDARD_RESIDUES:
            continue
        resname.append(rname)
        author_resnum.append(residue.get_id()[1])

        for atom in residue:
            if atom.is_disordered():
                atom = atom.disordered_get()
            altloc = atom.get_altloc()
            if altloc not in (" ", "A"):
                continue
            element = (atom.element or "").strip().upper()
            name = atom.get_name().strip()
            if element == "H" or name.startswith("H") or name.startswith("D"):
                continue

            atom_name.append(name)
            coord.append(atom.get_coord())
            atom_resindex.append(ridx)

            q = 0.0
            if rname in CHARGED_RESIDUES and name in CHARGE_ATOMS:
                q = charmag[CHARGE_ATOMS.index(name)]
            charge.append(q)
        ridx += 1

    if ridx == 0:
        raise ValueError("No standard-amino-acid residues found in selected chain")

    author_resnum = np.asarray(author_resnum, dtype=int)
    gaps = []
    diffs = np.diff(author_resnum)
    for i, d in enumerate(diffs):
        if d != 1:
            gaps.append((int(author_resnum[i]), int(author_resnum[i + 1])))
    if gaps:
        warnings.warn(
            f"Structure has {len(gaps)} residue-numbering gap(s) (missing/unmodeled "
            f"residues): {gaps}. The WSME model assumes a contiguous chain; residues "
            "are re-indexed sequentially by observed order, which is a reasonable "
            "approximation but treats the gap as if it had zero length.",
            stacklevel=2,
        )

    seq = "".join(RESNAME_TO_CODE[r] for r in resname)

    return Structure(
        resname=resname,
        seq=seq,
        author_resnum=author_resnum,
        atom_name=atom_name,
        coord=np.asarray(coord, dtype=float),
        atom_resindex=np.asarray(atom_resindex, dtype=int),
        charge=np.asarray(charge, dtype=float),
        chain_id=bio_chain.id,
        ph=ph,
        gaps=gaps,
    )
