"""Per-site pKa via Poisson-Boltzmann, with pairwise site-site coupling --
pipeline spec step 3.

Do NOT use model-compound pKa as the answer (per the pipeline spec) --
it is only the reference state a residue's *shift* is measured against.
The actual per-site intrinsic pKa is

    pKa_i = model_pKa_i - DeltaDeltaG_i / (RT ln10)

where DeltaDeltaG_i is the difference between (a) the electrostatic
energy of turning that site's charge from deprotonated to protonated
*in the protein/membrane environment* (all other titratable sites fixed
at their AMBER-default reference protonation state -- the standard
Bashford-Karplus "background" approximation) and (b) the same
protonation-energy difference for the isolated residue in a small model
compound in pure water. Site-site coupling W_ij (needed because the
histidine shell and buried carboxylates interact) is the interaction
energy between two sites' charge differences, isolated via a standard
double-difference of four whole-system PB energies -- this cancels each
site's own self-energy and any energy shared with the fixed background,
leaving only the i<->j interaction.

Charges for the protonated/deprotonated (or, for His, HIP-vs-HIE)
microstates come from PDB2PQR's own bundled AMBER ff99 charge set
(AMBER.DAT) -- real, versioned force-field data, not remembered/
fabricated numbers (see ``load_amber_charges``). Model-compound pKa
values reuse ``wsme_gpcr.structure.DEFAULT_PKA`` (already the working
reference table elsewhere in this repository) rather than introducing a
second, potentially inconsistent table.

Arg is excluded from active titration: PDB2PQR's AMBER.DAT has no neutral
guanidinium (ARN) variant (confirmed by inspection -- zero matching
entries), consistent with Arg's very high model pKa (~12.5) meaning it is
essentially never deprotonated across the pH 5-8 range this pipeline
scans. Its contribution to Delta_n_H is therefore always ~0 and it is
reported at its fixed model pKa rather than computed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from wsme_gpcr.structure import DEFAULT_PKA

R_KJ_PER_MOL_K = 8.31446261815324e-3
LN10 = np.log(10.0)

# Real AMBER ff99 protonated/deprotonated residue-name pairs (verified
# against PDB2PQR's own bundled AMBER.DAT -- see module docstring). His is
# treated as a 2-state HIP (protonated, both ring N's carry H) <-> HIE
# (neutral tautomer, H on NE2 only) system; the HID tautomer (H on ND1
# only) is a documented simplification not modeled here -- see
# ``compute_site_energy`` for how this affects the reported pKa.
TITRATABLE_RESIDUES = {
    "ASP": {"protonated_resname": "ASH", "deprotonated_resname": "ASP", "titratable_h": "HD2"},
    "GLU": {"protonated_resname": "GLH", "deprotonated_resname": "GLU", "titratable_h": "HE2"},
    "HIS": {"protonated_resname": "HIP", "deprotonated_resname": "HIE", "titratable_h": "HD1"},
    "LYS": {"protonated_resname": "LYS", "deprotonated_resname": "LYN", "titratable_h": "HZ1"},
}


def _amber_dat_path() -> Path:
    import pdb2pqr
    return Path(pdb2pqr.__file__).parent / "dat" / "AMBER.DAT"


def load_amber_charges() -> dict:
    """Parse PDB2PQR's bundled AMBER.DAT (AMBER ff99 partial charges) into
    {resname: {atomname: (charge, radius)}}."""
    charges = {}
    for line in _amber_dat_path().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        resname, atomname, charge, radius = parts[0], parts[1], parts[2], parts[3]
        charges.setdefault(resname, {})[atomname] = (float(charge), float(radius))
    return charges


@dataclass
class PqrAtom:
    serial: int
    name: str
    resname: str
    resnum: int
    x: float
    y: float
    z: float
    charge: float
    radius: float
    chain: str = "A"


def read_pqr(path) -> list:
    """Parse a PDB2PQR-format .pqr file (whitespace-delimited; PDB2PQR's
    own output omits the chain column, so both 9- and 10-field ATOM lines
    are accepted)."""
    atoms = []
    for line in Path(path).read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        parts = line.split()
        # parts[0]=ATOM, [1]=serial, [2]=name, [3]=resname, then either
        # [4]=resnum (no chain) or [4]=chain,[5]=resnum
        if parts[4].lstrip("-").isdigit():
            chain = "A"
            resnum, x, y, z, charge, radius = parts[4:10]
        else:
            chain = parts[4]
            resnum, x, y, z, charge, radius = parts[5:11]
        atoms.append(PqrAtom(
            serial=int(parts[1]), name=parts[2], resname=parts[3], chain=chain,
            resnum=int(resnum), x=float(x), y=float(y), z=float(z),
            charge=float(charge), radius=float(radius),
        ))
    return atoms


def write_pqr(path, atoms: list):
    lines = []
    for i, a in enumerate(atoms, start=1):
        lines.append(
            f"ATOM  {i:>5d} {a.name:<4s} {a.resname:<4s} {a.resnum:>4d}    "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f} {a.charge:7.4f} {a.radius:6.4f}"
        )
    lines.append("TER")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def build_microstate(base_atoms: list, resnum: int, resname: str, protonated: bool,
                      amber_charges: dict, extra_h_position=None) -> list:
    """Return a new atom list with residue ``resnum``'s charges/radii
    swapped to the AMBER protonated or deprotonated variant of ``resname``.
    All other residues are left exactly as given (the fixed-background
    approximation).

    ``extra_h_position`` supplies the 3D coordinate for the titratable H
    when building the *protonated* microstate and ``base_atoms`` doesn't
    already contain it (e.g. because the reference structure was prepared
    at a pH where this site defaults to deprotonated) -- typically taken
    from a structure prepared at low pH via ``structure_prep``.
    """
    info = TITRATABLE_RESIDUES[resname]
    target_resname = info["protonated_resname"] if protonated else info["deprotonated_resname"]
    h_name = info["titratable_h"]
    charge_table = amber_charges[target_resname]

    out = []
    h_atom = None
    last_atom_this_residue = None
    for atom in base_atoms:
        if atom.resnum != resnum:
            out.append(atom)
            continue
        last_atom_this_residue = atom
        if atom.name == h_name:
            h_atom = atom
            if not protonated:
                continue  # drop the titratable H for the deprotonated microstate
        if atom.name not in charge_table:
            raise KeyError(f"{target_resname} has no AMBER charge entry for atom {atom.name!r} "
                            f"(resnum {resnum}) -- base_atoms may not match the expected AMBER atom naming")
        charge, radius = charge_table[atom.name]
        out.append(replace(atom, charge=charge, radius=radius, resname=target_resname))

    if protonated and h_atom is None:
        if extra_h_position is None:
            raise ValueError(f"resnum {resnum} ({resname}) is missing its titratable H ({h_name}) "
                              f"in base_atoms and no extra_h_position was given to add it")
        if last_atom_this_residue is None:
            raise KeyError(f"resnum {resnum} not found in base_atoms")
        charge, radius = charge_table[h_name]
        out.append(PqrAtom(
            serial=-1, name=h_name, resname=target_resname, resnum=resnum,
            x=float(extra_h_position[0]), y=float(extra_h_position[1]), z=float(extra_h_position[2]),
            charge=charge, radius=radius, chain=last_atom_this_residue.chain,
        ))

    return out


def charge_delta(resname: str, amber_charges: dict) -> float:
    """Net charge change (protonated minus deprotonated) for one titratable
    residue type -- +1 for acids (Asp/Glu: deprotonated is anionic) and
    His/Lys (protonated is cationic), by construction of the AMBER charge
    sets; returned as the actual sum for a direct sanity check rather than
    assumed."""
    info = TITRATABLE_RESIDUES[resname]
    prot_total = sum(c for c, _ in amber_charges[info["protonated_resname"]].values())
    deprot_total = sum(c for c, _ in amber_charges[info["deprotonated_resname"]].values())
    return prot_total - deprot_total
