"""Classify a predicted GPCR model as active / inactive / intermediate using the
canonical class-A activation microswitch: the cytoplasmic TM3-TM6 distance ("ionic lock").

Background: in inactive-state class-A GPCRs, the conserved DRY-motif arginine (3.50) and
a residue near the cytoplasmic end of TM6 (~6.30-6.34) sit close together. Receptor
activation swings the cytoplasmic end of TM6 outward by ~8-14 Angstrom depending on the
receptor, breaking this contact (Deupi & Standfuss 2011; Latorraca et al. 2017; Zhou et
al. 2019). This single distance is a widely used, receptor-generic proxy for activation
state, but the exact residue numbers and the active/inactive distance values differ by
receptor -- see `calibrate_thresholds` below to fit them from reference structures instead
of trusting the generic defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ActivationThresholds:
    inactive_max: float
    active_min: float

    def classify(self, distance: float) -> str:
        if distance <= self.inactive_max:
            return "inactive"
        if distance >= self.active_min:
            return "active"
        return "intermediate"


# Generic class-A default, calibrated loosely from beta2AR/rhodopsin/M2R literature.
# Strongly prefer `calibrate_thresholds` with receptor-specific reference structures.
DEFAULT_THRESHOLDS = ActivationThresholds(inactive_max=10.5, active_min=13.0)


def tm3_tm6_distance(
    ca_coords: dict[int, np.ndarray], tm3_resnum: int, tm6_resnum: int
) -> float:
    """CA-CA distance (Angstrom) between the given TM3 and TM6 reference residues."""
    if tm3_resnum not in ca_coords or tm6_resnum not in ca_coords:
        missing = [r for r in (tm3_resnum, tm6_resnum) if r not in ca_coords]
        raise KeyError(f"Residue(s) {missing} not found in model (chain break or wrong numbering?)")
    return float(np.linalg.norm(ca_coords[tm3_resnum] - ca_coords[tm6_resnum]))


def calibrate_thresholds(
    inactive_ca_coords: dict[int, np.ndarray],
    active_ca_coords: dict[int, np.ndarray],
    tm3_resnum: int,
    tm6_resnum: int,
    margin_fraction: float = 0.15,
) -> ActivationThresholds:
    """Derive receptor-specific thresholds from one known inactive-state and one known
    active-state reference structure (e.g. an antagonist-bound and a G-protein/arrestin-
    bound PDB structure of the same receptor or a close homolog, with residues renumbered
    to match the target sequence). The band between the two reference distances, shrunk by
    `margin_fraction` on each side, is left as "intermediate"."""
    d_inactive = tm3_tm6_distance(inactive_ca_coords, tm3_resnum, tm6_resnum)
    d_active = tm3_tm6_distance(active_ca_coords, tm3_resnum, tm6_resnum)
    if d_active <= d_inactive:
        raise ValueError(
            f"Active-state distance ({d_active:.2f}) must exceed inactive-state "
            f"distance ({d_inactive:.2f}); check residue numbers and reference structures"
        )
    span = d_active - d_inactive
    margin = span * margin_fraction
    return ActivationThresholds(inactive_max=d_inactive + margin, active_min=d_active - margin)


def classify_model(
    ca_coords: dict[int, np.ndarray],
    tm3_resnum: int,
    tm6_resnum: int,
    thresholds: ActivationThresholds = DEFAULT_THRESHOLDS,
) -> tuple[str, float]:
    """Return (label, tm3_tm6_distance) for one model."""
    d = tm3_tm6_distance(ca_coords, tm3_resnum, tm6_resnum)
    return thresholds.classify(d), d


def mean_plddt(pdb_path: str | Path) -> float:
    """Mean per-residue pLDDT, read from the B-factor column that ColabFold/AF2 writes it
    into for CA atoms of chain A."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("m", str(pdb_path))
    values = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    values.append(residue["CA"].bfactor)
        break
    if not values:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return float(np.mean(values))


def passes_fold_quality(
    ca_coords: dict[int, np.ndarray], min_mean_plddt: float | None = None,
    mean_plddt_value: float | None = None, plddt_cutoff: float = 70.0,
    max_ca_bond_dev: float = 1.0, expected_ca_bond: float = 3.8,
) -> bool:
    """Sanity-check that a model is a plausible stable fold rather than an unfolded/garbage
    prediction from an overly aggressive MSA-subsampling run: (1) mean pLDDT above
    `plddt_cutoff`, and (2) consecutive-residue CA-CA distances close to the canonical
    3.8 Angstrom peptide-bond spacing (large deviations indicate chain breaks / extended,
    non-globular junk)."""
    if mean_plddt_value is not None and mean_plddt_value < plddt_cutoff:
        return False
    residues = sorted(ca_coords)
    for r1, r2 in zip(residues, residues[1:]):
        if r2 - r1 != 1:
            continue
        d = float(np.linalg.norm(ca_coords[r1] - ca_coords[r2]))
        if abs(d - expected_ca_bond) > max_ca_bond_dev:
            return False
    return True
