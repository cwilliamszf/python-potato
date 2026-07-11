"""Van der Waals contact map and electrostatic interaction list.

Ports the contact/electrostatics section of ``cmapCalcElecBlock.m``:
  - Short-range (VdW) contacts: heavy-atom pairs in different residues
    within ``vdw_cutoff`` Angstrom, excluding pairs where *both* atoms
    carry a titratable charge (those are handled electrostatically
    instead).
  - Long-range electrostatics: all heavy-atom pairs in different
    residues where *both* atoms carry a titratable charge, scored with
    a Coulomb term (Debye-Hueckel ionic-strength screening is applied
    later, at run time, since it depends on temperature).

The MATLAB code does this with an O(atoms^2) double loop; here it's
vectorized with a KDTree for the short-range cutoff search and dense
pairwise distances for the (much smaller) set of charged atoms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .structure import Structure

# 332 kcal*Angstrom/(mol*e^2) * 4.184 J/cal, divided by a fixed partial
# dielectric of 4 baked into the original formula (further ionic-strength
# / medium-dielectric screening is applied at run time in the WSME engine).
_COULOMB_CONST = 332.0 * 4.184 / 4.0


@dataclass
class ContactMap:
    nres: int
    srcont: np.ndarray  # (nres, nres) int, upper-triangular VdW contact counts
    elec_pairs: np.ndarray  # (K, 5): [resi, resj, dist, seqsep, energy_vacuum]


def compute_contact_map(structure: Structure, vdw_cutoff: float = 5.0, elec_cutoff: float = 1000.0,
                         exclude_atoms: np.ndarray = None) -> ContactMap:
    """``exclude_atoms``, if given, is a boolean mask (length natoms) of
    atoms to drop from all contact/electrostatic accounting -- e.g. the
    side-chain atoms of a computationally alanine-mutated residue (see
    alanine_scan.py). Excluded atoms simply can't participate in any
    contact or charged pair; they aren't removed from the Structure
    itself."""
    coord = structure.coord
    resindex = structure.atom_resindex
    charge = structure.charge
    nres = structure.nres

    if exclude_atoms is not None:
        charge = np.where(exclude_atoms, 0.0, charge)

    srcont = np.zeros((nres, nres), dtype=np.int64)

    if len(coord) >= 2:
        tree = cKDTree(coord)
        pairs = tree.query_pairs(r=vdw_cutoff, output_type="ndarray")
        if len(pairs):
            ri = resindex[pairs[:, 0]]
            rj = resindex[pairs[:, 1]]
            diff_res = ri != rj
            not_both_charged = (charge[pairs[:, 0]] == 0) | (charge[pairs[:, 1]] == 0)
            keep = diff_res & not_both_charged
            if exclude_atoms is not None:
                keep = keep & ~exclude_atoms[pairs[:, 0]] & ~exclude_atoms[pairs[:, 1]]
            lo = np.minimum(ri[keep], rj[keep])
            hi = np.maximum(ri[keep], rj[keep])
            np.add.at(srcont, (lo, hi), 1)

    charged_idx = np.where(charge != 0)[0]
    elec_rows = []
    if len(charged_idx) >= 2:
        cc = coord[charged_idx]
        cr = resindex[charged_idx]
        cq = charge[charged_idx]
        diff = cc[:, None, :] - cc[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        iu, ju = np.triu_indices(len(charged_idx), k=1)
        d = dist[iu, ju]
        ri, rj = cr[iu], cr[ju]
        diff_res = ri != rj
        within_cutoff = d <= elec_cutoff
        keep = diff_res & within_cutoff
        if np.any(keep):
            d_k = d[keep]
            ri_k, rj_k = ri[keep], rj[keep]
            qi_k, qj_k = cq[iu][keep], cq[ju][keep]
            energy = _COULOMB_CONST * qi_k * qj_k / d_k
            lo = np.minimum(ri_k, rj_k)
            hi = np.maximum(ri_k, rj_k)
            seqsep = np.abs(ri_k - rj_k)
            elec_rows = np.column_stack([lo, hi, d_k, seqsep, energy])

    elec_pairs = np.asarray(elec_rows, dtype=float) if len(elec_rows) else np.zeros((0, 5))

    return ContactMap(nres=nres, srcont=srcont, elec_pairs=elec_pairs)
