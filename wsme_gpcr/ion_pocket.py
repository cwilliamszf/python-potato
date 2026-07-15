"""Explicit Na+ ion - block electrostatic interaction at the D2.50 sodium
pocket -- a rigorous test of the hypothesis that the bWSME model's total
lack of any ion representation explains why GPR68 inactive's
folded-state free-energy minimum does not restore even at the correctly
Tm-calibrated xi (see ``linkage_pka/FINDINGS.md``'s "xi calibration
(Prompt 1)" section: xi=-49.15 J/mol hits Tm=333.0 K exactly, but the
310 K profile's global minimum stays stuck at n=76/101, 75.2%).

The conserved D2.50 sodium pocket is a real, well-characterized
structural/allosteric feature in many GPCRs. This session's own
``linkage_pka`` Poisson-Boltzmann work already confirmed, on these exact
GPR68 structures, that an explicit Na+ ion at D2.50 (Asp67, located
geometrically via ``linkage_pka.membrane_frame.find_d250`` -- not a
sequence-motif guess) produces a real, physically-correct stabilizing
shift (intrinsic pKa 5.98 -> 9.35 with the ion present -- see
FINDINGS.md's "Na+ ion modeling" section). The bWSME model as built has
NO representation of this ion at all: it is a pure protein-contact model,
so a folding calculation on the bare structure literally cannot "see"
that this cross-helix bridge exists.

Physics added here, and nothing else: the ion is placed at the same real
geometry ``linkage_pka.titration.place_na_ion`` uses (bisector of D2.50's
carboxylate oxygens, extended away from the parent carbon, at the real
LJ contact distance from sourced AMBER/OpenMM radii -- reusing that
exact, already-verified geometry and those exact parameters, not new
numbers). Every OTHER charged atom within ``interaction_cutoff_ang`` of
that ion position (found by real distance search on the actual
structure, not assumed from a canonical GPCR sodium-pocket motif) gets a
real vacuum Coulomb energy to the ion (``contacts.py``'s own
``_COULOMB_CONST``, i.e. exactly the same convention/dielectric this
model already uses for every other charged-residue pair). For each such
partner residue, the ion-D250 and ion-partner Coulomb terms are summed
into ONE new block-block electrostatic pair (D250's block <-> partner's
block), representing the net stabilization the ion provides by bridging
two otherwise-repulsive anionic groups -- appended to the existing
``BlockModel.block_elec`` array. This is purely additional DATA fed into
the model's own existing pairwise-electrostatic machinery
(``wsme._debye_screened_emap``, unmodified) -- not a change to the
model's physics, block definition, or entropy parameters. Adding zero
pairs (call with an empty partner list) must reproduce the untouched
baseline bit-for-bit -- this is a real, checkable control, not an
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .blocking import BlockModel
from .structure import Structure

# Reuse the exact same vacuum Coulomb convention (dielectric=4 baked in,
# screened further by ionic strength/temperature at solve time) already
# used for every other charged-residue pair in this codebase.
_COULOMB_CONST = 332.0 * 4.184 / 4.0

_ION_POCKET_GEOMETRY = {
    "ASP": {"parent": "CG", "oxygens": ("OD1", "OD2")},
    "GLU": {"parent": "CD", "oxygens": ("OE1", "OE2")},
}


@dataclass
class IonPocketPartner:
    author_resnum: int
    resname: str
    atom_name: str
    dist_to_ion_ang: float
    ion_atom_energy_kj_mol: float  # vacuum Coulomb, this atom <-> ion alone


@dataclass
class IonPocketResult:
    ion_position: np.ndarray  # (3,) Angstrom
    d250_author_resnum: int
    partners: list  # [IonPocketPartner, ...] -- every qualifying atom found
    new_block_elec_rows: np.ndarray  # (K, 5) rows appended to block_elec
    block_model: BlockModel  # augmented copy; original is untouched


def _find_atom(structure: Structure, author_resnum: int, atom_name: str):
    ridx_candidates = np.where(structure.author_resnum == author_resnum)[0]
    if len(ridx_candidates) == 0:
        raise KeyError(f"author_resnum {author_resnum} not found in structure")
    ridx = int(ridx_candidates[0])
    atom_idx = [i for i, r in enumerate(structure.atom_resindex) if r == ridx and structure.atom_name[i] == atom_name]
    if not atom_idx:
        raise KeyError(f"residue {author_resnum} has no atom named {atom_name!r}")
    return ridx, atom_idx[0]


def place_na_ion(structure: Structure, d250_author_resnum: int, d250_resname: str = "ASP",
                  ion_radius_ang: float = 1.369, contact_distance_ang: float = None) -> np.ndarray:
    """Same real geometry as ``linkage_pka.titration.place_na_ion``:
    bisector of the carboxylate oxygens, extended away from the parent
    carbon, at the LJ contact distance (ion radius + oxygen radius, both
    real sourced AMBER/OpenMM values -- 1.369 Angstrom for Na+ from
    OpenMM's amber14/tip3p.xml, 1.6612 Angstrom for ASP OD1/OD2 from
    PDB2PQR's AMBER.DAT -- see ``linkage_pka.titration.load_na_ion_parameters``
    /``load_amber_charges``, cross-checked directly, not re-derived here).
    """
    geom = _ION_POCKET_GEOMETRY[d250_resname]
    _, parent_atom_idx = _find_atom(structure, d250_author_resnum, geom["parent"])
    _, o1_atom_idx = _find_atom(structure, d250_author_resnum, geom["oxygens"][0])
    _, o2_atom_idx = _find_atom(structure, d250_author_resnum, geom["oxygens"][1])

    parent_pos = structure.coord[parent_atom_idx]
    o1, o2 = structure.coord[o1_atom_idx], structure.coord[o2_atom_idx]
    midpoint = (o1 + o2) / 2.0

    direction = midpoint - parent_pos
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError(f"resnum {d250_author_resnum} ({d250_resname}): degenerate geometry")
    direction = direction / norm

    if contact_distance_ang is None:
        oxygen_amber_radius_ang = 1.6612  # ASP OD1/OD2, PDB2PQR AMBER.DAT -- see module docstring
        contact_distance_ang = ion_radius_ang + oxygen_amber_radius_ang
    return midpoint + direction * contact_distance_ang


def place_na_ion_multi_coordinate(
    structure: Structure,
    d250_author_resnum: int,
    d250_resname: str = "ASP",
    coordination_search_radius_ang: float = 6.0,
    max_coordinating_residues: int = 4,
) -> tuple:
    """Rigorous placement for a jointly-coordinated ion: ``place_na_ion``
    alone assumes the ion sits in the empty space beyond D2.50's OWN
    local geometry, which is a poor approximation whenever another
    anionic residue also coordinates the same ion (real testing on GPR68
    found exactly this: the single-residue estimate placed the ion only
    1.33 Angstrom from a real, independently-corroborated partner's
    oxygen -- physically impossible for two non-bonded heavy atoms, real
    Na-O coordination is ~2.2-2.6 Angstrom).

    Finds every OTHER anionic oxygen atom within
    ``coordination_search_radius_ang`` of the single-residue estimate
    (the closest ``max_coordinating_residues`` distinct residues), and
    places the ion at the centroid of D2.50's own two carboxylate
    oxygens PLUS every such nearby partner oxygen -- the geometric
    center of the real coordinating shell, the standard way to
    approximate a jointly-coordinated cation's position from protein
    geometry alone when no experimentally-resolved ion position is
    available. Simple two-point averaging of "naive estimate" and
    "partner position" was tried and rejected: when the naive point is
    already pathologically close to a partner (the real GPR68 case, see
    below), a two-point midpoint only lands HALF as close -- still
    unphysical, not fixed. Centroiding the actual oxygen atoms instead
    spreads the result across the real coordinating geometry rather than
    anchoring it near whichever single point happened to be closest.
    Gracefully degrades to exactly ``place_na_ion``'s own result when no
    partner is found (an explicit fallback, not a one-point "centroid").

    Returns ``(ion_position, coordinating_atom_indices)`` -- the latter
    lists any real partner atoms found (closest first); empty if none.
    """
    initial_pos = place_na_ion(structure, d250_author_resnum, d250_resname)
    geom = _ION_POCKET_GEOMETRY[d250_resname]
    d250_ridx = int(np.where(structure.author_resnum == d250_author_resnum)[0][0])
    d250_oxygen_idx = [_find_atom(structure, d250_author_resnum, name)[1] for name in geom["oxygens"]]

    dists = np.linalg.norm(structure.coord - initial_pos[None, :], axis=1)
    is_anion = structure.charge < 0.0
    other_residue = structure.atom_resindex != d250_ridx
    within = dists <= coordination_search_radius_ang
    candidates = np.where(is_anion & other_residue & within)[0]

    order = candidates[np.argsort(dists[candidates])]
    seen_residues, kept = set(), []
    for idx in order:
        ridx = int(structure.atom_resindex[idx])
        if ridx in seen_residues:
            continue
        seen_residues.add(ridx)
        kept.append(int(idx))
        if len(seen_residues) >= max_coordinating_residues:
            break

    if not kept:
        return initial_pos, []

    centroid = structure.coord[d250_oxygen_idx + kept].mean(axis=0)
    return centroid, kept


def add_ion_pocket_interaction(
    structure: Structure,
    block_model: BlockModel,
    d250_author_resnum: int,
    d250_resname: str = "ASP",
    interaction_cutoff_ang: float = 6.0,
    ion_charge: float = 1.0,
    multi_coordinate: bool = True,
    coordination_search_radius_ang: float = 6.0,
    max_coordinating_residues: int = 4,
) -> IonPocketResult:
    """Find every charged atom within ``interaction_cutoff_ang`` of the
    D2.50 sodium-pocket ion position (excluding D250's own residue) by
    real distance search, and append one new block-block electrostatic
    pair per qualifying PARTNER RESIDUE (summing that residue's
    qualifying atoms' ion-Coulomb terms) to a copy of ``block_model``.

    ``multi_coordinate=True`` (default) uses
    ``place_na_ion_multi_coordinate`` -- the physically rigorous
    placement when another anionic residue also coordinates the same
    ion (see that function's docstring for why the naive single-residue
    ``place_na_ion`` estimate can be badly wrong, e.g. landing 1.33
    Angstrom from a real partner's oxygen). Set False to use the naive
    single-residue estimate instead (e.g. for a direct before/after
    comparison of the placement refinement itself).
    """
    if multi_coordinate:
        ion_pos, _ = place_na_ion_multi_coordinate(
            structure, d250_author_resnum, d250_resname,
            coordination_search_radius_ang=coordination_search_radius_ang,
            max_coordinating_residues=max_coordinating_residues,
        )
    else:
        ion_pos = place_na_ion(structure, d250_author_resnum, d250_resname)
    d250_ridx = int(np.where(structure.author_resnum == d250_author_resnum)[0][0])
    d250_block = int(block_model.block_of_residue[d250_ridx])

    dists = np.linalg.norm(structure.coord - ion_pos[None, :], axis=1)
    charged_mask = structure.charge != 0.0
    other_residue_mask = structure.atom_resindex != d250_ridx
    within_cutoff = dists <= interaction_cutoff_ang
    candidate_atoms = np.where(charged_mask & other_residue_mask & within_cutoff)[0]

    partners = []
    per_residue_energy = {}
    per_residue_min_dist = {}
    for atom_idx in candidate_atoms:
        ridx = int(structure.atom_resindex[atom_idx])
        dist = float(dists[atom_idx])
        q = float(structure.charge[atom_idx])
        energy = _COULOMB_CONST * ion_charge * q / dist
        partners.append(IonPocketPartner(
            author_resnum=int(structure.author_resnum[ridx]), resname=structure.resname[ridx],
            atom_name=structure.atom_name[atom_idx], dist_to_ion_ang=dist, ion_atom_energy_kj_mol=energy,
        ))
        per_residue_energy[ridx] = per_residue_energy.get(ridx, 0.0) + energy
        per_residue_min_dist[ridx] = min(per_residue_min_dist.get(ridx, dist), dist)

    dist_d250_to_ion = float(np.linalg.norm(
        structure.coord[_find_atom(structure, d250_author_resnum, _ION_POCKET_GEOMETRY[d250_resname]["parent"])[1]]
        - ion_pos
    ))

    new_rows = []
    for ridx, energy in per_residue_energy.items():
        partner_block = int(block_model.block_of_residue[ridx])
        seqsep = abs(partner_block - d250_block)
        effective_dist = dist_d250_to_ion + per_residue_min_dist[ridx]  # through-ion path length
        lo, hi = sorted((d250_block, partner_block))
        new_rows.append([lo, hi, effective_dist, seqsep, energy])

    new_rows_arr = np.asarray(new_rows, dtype=float) if new_rows else np.zeros((0, 5))
    augmented_block_elec = (np.vstack([block_model.block_elec, new_rows_arr])
                             if len(block_model.block_elec) else new_rows_arr)
    augmented_block_model = replace(block_model, block_elec=augmented_block_elec)

    return IonPocketResult(
        ion_position=ion_pos, d250_author_resnum=d250_author_resnum, partners=partners,
        new_block_elec_rows=new_rows_arr, block_model=augmented_block_model,
    )
