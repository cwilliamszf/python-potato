"""
Collective variables (CVs) for GPCR conformational landscapes.

Reproduces the four microswitch distance types used in Fleetwood et al.
2021 (Figure 1a): a closest-heavy-atom distance (TM5 bulge, ionic lock), an
atom-atom distance (the Y-Y motif's C-zeta / C-zeta distance) and a
"connector Delta-RMSD" (RMSD to an active-like reference minus RMSD to an
inactive-like reference, for a small set of connector-region residues).

CVs are defined generically by chain id + residue number so this works for
any GPCR ensemble, not just beta2AR -- the paper's own residue numbers are
provided in :data:`BETA2AR_MICROSWITCHES` purely as a usable example/default.

A residue selector is a dict: ``{"chain": "A", "resid": 207}``.
An atom selector adds an atom name: ``{"chain": "A", "resid": 219, "atom": "CZ"}``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from Bio.PDB.Structure import Structure


def _get_residue(structure: Structure, chain_id: str, resid: int, model_index: int = 0):
    model = structure[model_index]
    chain = model[chain_id]
    for residue in chain:
        if residue.id[1] == resid:
            return residue
    raise KeyError(f"Residue {resid} not found in chain '{chain_id}' of structure '{structure.id}'")


def _heavy_atoms(residue):
    return [atom for atom in residue if atom.element != "H"]


def closest_heavy_atom_distance(
    structure: Structure, sel1: Dict, sel2: Dict, model_index: int = 0
) -> float:
    """Closest heavy-atom distance (nm... or whatever unit the coordinates use,
    typically Angstrom for PDB files) between two residues.

    Matches the paper's definition of the TM5 bulge and ionic lock CVs.
    """
    res1 = _get_residue(structure, sel1["chain"], sel1["resid"], model_index)
    res2 = _get_residue(structure, sel2["chain"], sel2["resid"], model_index)
    atoms1 = _heavy_atoms(res1)
    atoms2 = _heavy_atoms(res2)
    if not atoms1 or not atoms2:
        raise ValueError("One of the selected residues has no heavy atoms")
    coords1 = np.array([a.coord for a in atoms1])
    coords2 = np.array([a.coord for a in atoms2])
    diffs = coords1[:, None, :] - coords2[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return float(dists.min())


def atom_distance(structure: Structure, sel1: Dict, sel2: Dict, model_index: int = 0) -> float:
    """Distance between two named atoms, e.g. the Y-Y motif's CZ-CZ distance."""
    res1 = _get_residue(structure, sel1["chain"], sel1["resid"], model_index)
    res2 = _get_residue(structure, sel2["chain"], sel2["resid"], model_index)
    atom1 = res1[sel1["atom"]]
    atom2 = res2[sel2["atom"]]
    return float(np.linalg.norm(atom1.coord - atom2.coord))


def _residue_atom_coords(residue, atom_names: Optional[Sequence[str]] = None) -> np.ndarray:
    coords = [atom.coord for atom in residue if atom_names is None or atom.get_name() in atom_names]
    if not coords:
        raise ValueError(f"No matching atoms found in residue {residue.id}")
    return np.array(coords)


def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Plain (no superposition) RMSD between two matching coordinate arrays."""
    coords_a = np.asarray(coords_a)
    coords_b = np.asarray(coords_b)
    if coords_a.shape != coords_b.shape:
        raise ValueError(
            f"Coordinate sets must have matching shape/atom order for RMSD, "
            f"got {coords_a.shape} vs {coords_b.shape}"
        )
    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def connector_delta_rmsd(
    structure: Structure,
    active_ref: Structure,
    inactive_ref: Structure,
    residues: Sequence[int],
    chain: str = "A",
    atom_names: Optional[Sequence[str]] = None,
    model_index: int = 0,
) -> float:
    """RMSD(structure, active_ref) - RMSD(structure, inactive_ref) over the
    given residues, i.e. the paper's "connector Delta-RMSD" CV: negative
    values are active-like, positive values are inactive-like.

    ``active_ref``/``inactive_ref`` must already be aligned into the same
    frame as ``structure`` (e.g. all conformers generated from/aligned to
    the same reference model), since no superposition is performed here.
    """

    def gather(struct: Structure) -> np.ndarray:
        parts = [
            _residue_atom_coords(_get_residue(struct, chain, resid, model_index), atom_names)
            for resid in residues
        ]
        return np.concatenate(parts, axis=0)

    query = gather(structure)
    active = gather(active_ref)
    inactive = gather(inactive_ref)
    return rmsd(query, active) - rmsd(query, inactive)


def evaluate_cv(structure: Structure, cv_def: Dict, model_index: int = 0, refs: Optional[Dict[str, Structure]] = None) -> float:
    """Evaluate a single CV definition (see the ``*_MICROSWITCHES`` examples
    below for the expected schema) against one structure."""
    kind = cv_def["type"]
    if kind == "closest_heavy_distance":
        return closest_heavy_atom_distance(structure, cv_def["sel1"], cv_def["sel2"], model_index)
    if kind == "atom_distance":
        return atom_distance(structure, cv_def["sel1"], cv_def["sel2"], model_index)
    if kind == "connector_delta_rmsd":
        if refs is None:
            raise ValueError(f"CV '{cv_def['name']}' requires `refs` (active/inactive reference structures)")
        active_ref = refs[cv_def["active_ref"]]
        inactive_ref = refs[cv_def["inactive_ref"]]
        return connector_delta_rmsd(
            structure,
            active_ref,
            inactive_ref,
            cv_def["residues"],
            chain=cv_def.get("chain", "A"),
            atom_names=cv_def.get("atom_names"),
            model_index=model_index,
        )
    raise ValueError(f"Unknown CV type: '{kind}'")


def evaluate_cvs(
    structure: Structure, cv_defs: List[Dict], model_index: int = 0, refs: Optional[Dict[str, Structure]] = None
) -> Dict[str, float]:
    """Evaluate a list of CV definitions against one structure."""
    return {cv_def["name"]: evaluate_cv(structure, cv_def, model_index, refs) for cv_def in cv_defs}


# Example CV set matching Figure 1a/2 of Fleetwood et al. 2021 for beta2AR
# (Ballesteros-Weinstein numbers in comments; residue numbers are beta2AR
# sequence numbers, chain "A" as typically used in single-chain PDB files).
# `connector_delta_rmsd` entries additionally require `refs={"active": ...,
# "inactive": ...}` (e.g. PDB 3P0G and 2RH1) when evaluated.
BETA2AR_MICROSWITCHES: List[Dict] = [
    {
        "name": "tm5_bulge",
        "type": "closest_heavy_distance",
        "sel1": {"chain": "A", "resid": 207},  # S207 (5.46)
        "sel2": {"chain": "A", "resid": 315},  # G315 (7.41)
    },
    {
        "name": "ionic_lock",
        "type": "closest_heavy_distance",
        "sel1": {"chain": "A", "resid": 268},  # E268 (6.30)
        "sel2": {"chain": "A", "resid": 131},  # R131 (3.50)
    },
    {
        "name": "y_y_motif",
        "type": "atom_distance",
        "sel1": {"chain": "A", "resid": 219, "atom": "CZ"},  # Y219 (5.58)
        "sel2": {"chain": "A", "resid": 326, "atom": "CZ"},  # Y326 (7.53)
    },
    {
        "name": "connector_drmsd",
        "type": "connector_delta_rmsd",
        "residues": [121, 282],  # I121 (3.40), F282 (6.44)
        "chain": "A",
        "active_ref": "active",
        "inactive_ref": "inactive",
    },
]
