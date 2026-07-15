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

# Residues that carry a titratable side-chain charge, the atoms across
# which the fully-charged unit charge is distributed, and their default
# (free-amino-acid) pKa. The original cmapCalcElecBlock.m used a fixed
# 4-tier lookup table (pH 7/5/3.5/2) with hand-picked charge magnitudes;
# here that's replaced with a continuous Henderson-Hasselbalch titration
# so arbitrary/fine-grained pH values (and per-residue pKa overrides, for
# a specific residue known/suspected to have an environment-shifted pKa
# -- e.g. a proposed GPCR proton sensor) can be used directly. At the
# original table's four anchor pH values this reproduces very similar
# (not bit-identical) charges: e.g. His at pH 7 is ~9% protonated here
# rather than the original's hard 0%, which is the physically more
# accurate answer for a residue with pKa exactly 6, not a deviation.
ACIDIC_RESIDUES = {"ASP", "GLU"}  # charged (deprotonated) above their pKa
BASIC_RESIDUES = {"HIS", "LYS", "ARG"}  # charged (protonated) below their pKa
CHARGED_RESIDUES = ACIDIC_RESIDUES | BASIC_RESIDUES

DEFAULT_PKA = {
    "ASP": 3.9,
    "GLU": 4.1,
    "HIS": 6.0,
    "LYS": 10.5,
    "ARG": 12.5,
}

# Per-atom charge at full (100%) ionization; split evenly across the atoms
# that share the group's formal charge, matching cmapCalcElecBlock.m.
FULL_CHARGE_ATOMS = {
    "ASP": {"OD1": -0.5, "OD2": -0.5},
    "GLU": {"OE1": -0.5, "OE2": -0.5},
    "HIS": {"ND1": 0.5, "NE2": 0.5},
    "LYS": {"NZ": 1.0},
    "ARG": {"NE": 0.33, "NH1": 0.33, "NH2": 0.33},
}


def fraction_charged(ph: float, pka: float, resname: str) -> float:
    """Henderson-Hasselbalch fraction of the group in its charged state.

    Acidic groups (Asp/Glu) are charged when deprotonated, i.e. above
    their pKa; basic groups (His/Lys/Arg) are charged when protonated,
    i.e. below their pKa.
    """
    if resname in ACIDIC_RESIDUES:
        return 1.0 / (1.0 + 10.0 ** (pka - ph))
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


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
    bfactor: np.ndarray  # (natoms,) B-factor/pLDDT column verbatim from the PDB/mmCIF file
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


def load_structure(path, chain: str | None = None, model: int = 0, ph: float = 7.0,
                    pka_overrides: dict | None = None) -> Structure:
    """Load a PDB or mmCIF file into a curated heavy-atom Structure.

    Only ATOM records for the 20 standard amino acids are kept; hydrogens,
    waters, ligands, and alternate (non-primary) conformers are dropped.

    Charges are assigned by continuous Henderson-Hasselbalch titration
    (see ``fraction_charged``) using ``DEFAULT_PKA`` per residue type,
    unless overridden. ``pka_overrides`` maps an *author* residue number
    to a custom pKa -- e.g. for a specific histidine proposed to have an
    environment-shifted pKa (a candidate pH-sensor residue).
    """
    from Bio.PDB import MMCIFParser, PDBParser

    pka_overrides = pka_overrides or {}
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
    atom_name, coord, atom_resindex, charge, bfactor = [], [], [], [], []

    ridx = 0
    for residue in bio_chain:
        hetflag, _, _ = residue.get_id()
        rname = residue.get_resname()
        if hetflag != " " or rname not in STANDARD_RESIDUES:
            continue
        resname.append(rname)
        this_author_resnum = residue.get_id()[1]
        author_resnum.append(this_author_resnum)

        group_charge = None
        if rname in CHARGED_RESIDUES:
            pka = pka_overrides.get(this_author_resnum, DEFAULT_PKA[rname])
            group_charge = FULL_CHARGE_ATOMS[rname]
            frac = fraction_charged(ph, pka, rname)

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
            bfactor.append(atom.get_bfactor())

            q = 0.0
            if group_charge is not None and name in group_charge:
                q = group_charge[name] * frac
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
        bfactor=np.asarray(bfactor, dtype=float),
        chain_id=bio_chain.id,
        ph=ph,
        gaps=gaps,
    )
