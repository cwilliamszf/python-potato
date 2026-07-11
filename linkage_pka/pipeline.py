"""End-to-end per-structure titration: identify titratable sites on a
prepared structure, compute each site's intrinsic pKa and the pairwise
couplings between nearby sites (pipeline spec step 3), solve the coupled
multi-site titration (``multisite.py``), and -- given a second structure
for the other conformer -- combine into the activation linkage
observables (``linkage.py``) that answer the pipeline's actual question.

This module is the orchestration glue between ``titration.py`` (single-site
and single-pair PB energies), ``multisite.py`` (coupled Boltzmann
titration), and ``linkage.py`` (Wyman/Tanford activation thermodynamics).
It performs no new physics of its own.

Cost note: every site costs 2 full PB solve pairs (protein + model
compound, each itself a solvated/reference Born-cycle pair -- see
``titration.compute_solvation_energy``), and every coupled pair costs 4
more whole-system solves (``compute_pairwise_coupling``). A real GPCR has
dozens of titratable sites; an unrestricted all-pairs coupling scan is
O(n^2) in the expensive step. ``coupling_distance_cutoff_ang`` exists to
keep this tractable by only computing coupling for pairs close enough to
plausibly interact -- it is a computational-cost cutoff, not a claim that
electrostatics vanishes beyond it (Poisson-Boltzmann with a real ionic
strength screens interactions on a similar length scale, which is why this
is a defensible approximation, but it is still an approximation and is
reported as a run parameter, not hidden).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .linkage import delta_g_act_from_ln_z, delta_n_h_from_theta
from .multisite import (
    DEFAULT_COUPLING_THRESHOLD_KJ_MOL,
    MultiSiteTitrationResult,
    solve_titration,
)
from .titration import (
    GridParams,
    compute_intrinsic_pka,
    compute_pairwise_coupling,
    load_amber_charges,
    place_titratable_hydrogen,
)

# Maps every AMBER protonation-state variant resname (whatever the base
# structure happens to carry, e.g. PDB2PQR's default-pH assignment) back to
# the canonical titratable residue type used throughout titration.py/
# multisite.py. Deliberately excludes ARG: PDB2PQR's AMBER.DAT has no
# neutral guanidinium (ARN) variant, so it cannot be titrated by this
# pipeline's build_microstate machinery -- see titration.py's module
# docstring for the full justification (Arg's model pKa ~12.5 means its
# Delta_n_H contribution across pH 5-8 is ~0 regardless).
_CANONICAL_RESNAME = {
    "ASP": "ASP", "ASH": "ASP",
    "GLU": "GLU", "GLH": "GLU",
    "HIS": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "LYS": "LYS", "LYN": "LYS",
}


def identify_titratable_sites(atoms: list) -> list:
    """Every (resnum, canonical_resname) titratable site present in
    ``atoms``, sorted by resnum. Canonicalizes whatever protonation-state
    variant the base structure happens to carry (e.g. HIE vs HIP) back to
    the residue type ``compute_intrinsic_pka``/``build_microstate`` expect;
    a residue's actual base protonation state doesn't matter here since
    both microstates are built explicitly regardless of the starting state.
    """
    by_resnum = {}
    for a in atoms:
        by_resnum.setdefault(a.resnum, a.resname)
    sites = [
        (resnum, _CANONICAL_RESNAME[resname])
        for resnum, resname in by_resnum.items()
        if resname in _CANONICAL_RESNAME
    ]
    return sorted(sites)


def _residue_coords(atoms: list, resnum: int) -> np.ndarray:
    coords = np.array([[a.x, a.y, a.z] for a in atoms if a.resnum == resnum])
    if coords.size == 0:
        raise KeyError(f"resnum {resnum} not found in atoms")
    return coords


def residue_min_distance(atoms: list, resnum_i: int, resnum_j: int) -> float:
    """Minimum atom-atom Euclidean distance (Angstrom) between two
    residues' atom sets -- closest-approach distance, not centroid
    distance, so it correctly flags residues whose side chains reach
    toward each other even with distant backbones."""
    coords_i = _residue_coords(atoms, resnum_i)
    coords_j = _residue_coords(atoms, resnum_j)
    return float(cdist(coords_i, coords_j).min())


def find_coupled_pairs(atoms: list, sites: list, distance_cutoff_ang: float = 10.0) -> list:
    """All (site_i, site_j) pairs from ``sites`` (as returned by
    ``identify_titratable_sites``) whose closest-approach distance is
    within ``distance_cutoff_ang`` -- the candidate set to actually run
    ``compute_pairwise_coupling`` on. Sites farther apart are treated as
    uncoupled (W_ij=0) without spending an APBS call to confirm it."""
    pairs = []
    for idx_i in range(len(sites)):
        for idx_j in range(idx_i + 1, len(sites)):
            site_i, site_j = sites[idx_i], sites[idx_j]
            if residue_min_distance(atoms, site_i[0], site_j[0]) <= distance_cutoff_ang:
                pairs.append((site_i, site_j))
    return pairs


@dataclass
class StructureTitrationResult:
    ph: np.ndarray
    sites: list                    # [(resnum, canonical_resname), ...]
    site_energies: dict             # resnum -> SiteEnergyResult
    pka_intrinsic: dict             # resnum -> float
    coupling: dict                  # (resnum_i, resnum_j) -> W_ij kJ/mol
    coupled_pairs: list              # [((resnum_i,resname_i),(resnum_j,resname_j)), ...] actually computed
    multisite: MultiSiteTitrationResult


def run_structure_titration(
    atoms: list,
    frame,
    protein_grid_params: GridParams,
    model_grid_params: GridParams,
    ph_values,
    work_dir,
    sites: list = None,
    coupling_distance_cutoff_ang: float = 10.0,
    coupling_threshold_kj_mol: float = DEFAULT_COUPLING_THRESHOLD_KJ_MOL,
    membrane_dielectric: float = 2.0,
    amber_charges: dict = None,
    temp_k: float = 298.15,
) -> StructureTitrationResult:
    """Full per-structure titration: every site's intrinsic pKa, coupling
    for every nearby pair, and the resulting coupled multi-site theta(pH)
    -- one fixed structure (active or inactive), pipeline spec step 3+5.

    ``sites`` defaults to ``identify_titratable_sites(atoms)`` (every
    titratable residue on the structure); pass an explicit subset to
    restrict a run (e.g. for testing, or to focus compute on a known
    cluster).
    """
    amber_charges = amber_charges or load_amber_charges()
    work_dir = Path(work_dir)
    sites = sites if sites is not None else identify_titratable_sites(atoms)
    ph_values = np.asarray(list(ph_values), dtype=float)

    h_positions = {resnum: place_titratable_hydrogen(atoms, resnum, resname) for resnum, resname in sites}

    site_energies = {}
    pka_intrinsic = {}
    for resnum, resname in sites:
        result = compute_intrinsic_pka(
            atoms, resnum, resname, frame,
            protein_grid_params, model_grid_params, work_dir / f"site_{resnum}",
            amber_charges=amber_charges, membrane_dielectric=membrane_dielectric,
            temp_k=temp_k, extra_h_position=h_positions[resnum],
        )
        site_energies[resnum] = result
        pka_intrinsic[resnum] = result.intrinsic_pka

    coupled_pairs = find_coupled_pairs(atoms, sites, coupling_distance_cutoff_ang)
    coupling = {}
    for (resnum_i, resname_i), (resnum_j, resname_j) in coupled_pairs:
        w_ij = compute_pairwise_coupling(
            atoms, resnum_i, resname_i, resnum_j, resname_j, frame, protein_grid_params,
            work_dir / f"coupling_{resnum_i}_{resnum_j}", amber_charges=amber_charges,
            membrane_dielectric=membrane_dielectric,
            extra_h_position_i=h_positions[resnum_i], extra_h_position_j=h_positions[resnum_j],
        )
        coupling[(resnum_i, resnum_j)] = w_ij

    multisite = solve_titration(pka_intrinsic, coupling, ph_values, coupling_threshold_kj_mol, temp_k)

    return StructureTitrationResult(
        ph=ph_values, sites=sites, site_energies=site_energies, pka_intrinsic=pka_intrinsic,
        coupling=coupling, coupled_pairs=coupled_pairs, multisite=multisite,
    )


@dataclass
class ActivationLinkageResult:
    ph: np.ndarray
    delta_g_act: np.ndarray
    delta_n_h: np.ndarray
    delta_n_h_per_residue: np.ndarray
    resnums: np.ndarray

    def top_contributors(self, ph_value: float, n: int = 10) -> list:
        """Same convention as ``linkage.LinkageResult.top_contributors``."""
        i = int(np.argmin(np.abs(self.ph - ph_value)))
        contrib = self.delta_n_h_per_residue[i]
        order = np.argsort(-np.abs(contrib))[:n]
        return [(int(self.resnums[j]), float(contrib[j])) for j in order]


def compute_activation_linkage(
    active: StructureTitrationResult,
    inactive: StructureTitrationResult,
    T: float = 298.15,
) -> ActivationLinkageResult:
    """DeltaDeltaG_act(pH) and Delta_n_H(pH) between two
    ``StructureTitrationResult``s (active/inactive), via the coupled-
    titration generalizations in ``linkage.py``
    (``delta_n_h_from_theta``/``delta_g_act_from_ln_z``) -- the coupled
    analogue of ``linkage.compute_linkage``, valid whether or not either
    structure has any coupled clusters."""
    if not np.array_equal(active.ph, inactive.ph):
        raise ValueError("active and inactive StructureTitrationResults must share the same pH grid")

    resnums, per_residue, delta_n_h = delta_n_h_from_theta(active.multisite.theta, inactive.multisite.theta)
    delta_g_act = delta_g_act_from_ln_z(active.multisite.ln_z_total, inactive.multisite.ln_z_total, T=T)

    return ActivationLinkageResult(
        ph=active.ph, delta_g_act=delta_g_act, delta_n_h=delta_n_h,
        delta_n_h_per_residue=per_residue, resnums=resnums,
    )
