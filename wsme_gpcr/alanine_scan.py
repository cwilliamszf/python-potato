"""In silico alanine-scanning mutagenesis, replicating the protocol in
Anantakrishnan & Naganathan, "Thermodynamic architecture and
conformational plasticity of GPCRs," Nat Commun 14, 128 (2023), Fig. 7.

Protocol (from the paper): for each mutation site, computationally
truncate that residue's side chain to alanine (only N/CA/C/O/CB atoms
remain -- no atomic detail beyond that, matching the Go-model contact
accounting the WSME model already uses), recompute the ensemble, and
compare its *positive* coupling free energy matrix (chi_plus, "DeltaG+"
in the paper -- the states that harbor coupled residues in the folded
ensemble) against the wild-type matrix. The element-wise difference
(Delta-Delta-G+) is averaged over one axis to give a per-block vector
describing how much every block's average coupling shifted due to that
one mutation; stacking those vectors over many mutation sites and taking
their mean/std gives the "mutational response" (MR).

Mutating a residue never changes secondary structure, hence never
changes the block partition (block boundaries come only from
secondary-structure runs + block_size) -- so a mutant's chi_plus is
always directly, element-wise comparable to the wild type's, no
re-alignment needed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .blocking import BlockModel, build_blocks
from .contacts import compute_contact_map
from .coupling import compute_coupling
from .structure import Structure
from .wsme import WSMEParams

# Atoms kept after a computational Ala mutation (matches PyMOL's
# mutagenesis wizard truncation: backbone + CB, everything else removed).
BACKBONE_PLUS_CB = {"N", "CA", "C", "O", "CB"}
EXCLUDED_FROM_SCAN = {"ALA", "GLY", "PRO"}  # no meaningful Ala mutation / paper excludes these


def alanine_exclude_mask(structure: Structure, resnums) -> np.ndarray:
    """Boolean atom mask: True for side-chain atoms (beyond CB) of the
    given author residue numbers -- pass to compute_contact_map's
    exclude_atoms to computationally mutate those residues to alanine."""
    resnums = set(int(r) for r in resnums)
    atom_resnum = structure.author_resnum[structure.atom_resindex]
    is_mutated_res = np.isin(atom_resnum, list(resnums))
    is_sidechain = np.array([name not in BACKBONE_PLUS_CB for name in structure.atom_name])
    return is_mutated_res & is_sidechain


def scannable_positions(structure: Structure) -> list:
    """Author resnums eligible for alanine scanning (excludes existing
    Ala/Gly/Pro), matching the paper's site selection."""
    return [
        int(structure.author_resnum[i])
        for i in range(structure.nres)
        if structure.resname[i] not in EXCLUDED_FROM_SCAN
    ]


def mutant_chi_plus(structure: Structure, ss_mask: np.ndarray, block_size: int, params: WSMEParams, resnums) -> np.ndarray:
    """chi_plus (positive coupling free energy matrix) for the structure
    with the given residue(s) computationally mutated to alanine."""
    exclude = alanine_exclude_mask(structure, resnums)
    cm = compute_contact_map(structure, exclude_atoms=exclude)
    bm = build_blocks(ss_mask, cm, block_size=block_size)
    return compute_coupling(structure, bm, ss_mask, params).chi_plus, bm


@dataclass
class AlanineScanResult:
    positions: list  # author resnums scanned
    block_of_position: dict  # resnum -> block index
    wt_chi_plus: np.ndarray  # (nblocks, nblocks)
    ddg_plus: dict = field(default_factory=dict)  # resnum -> (nblocks, nblocks) chi_plus_mut - chi_plus_wt
    mean_ddg_vector: dict = field(default_factory=dict)  # resnum -> (nblocks,) row-averaged DeltaDeltaG+
    MR_mean: np.ndarray = None  # (nblocks,) mean mutational response across all scanned positions
    MR_std: np.ndarray = None  # (nblocks,) std of mutational response


def run_alanine_scan(
    structure: Structure,
    ss_mask: np.ndarray,
    params: WSMEParams,
    positions,
    block_size: int = 4,
    wt_block_model: BlockModel = None,
    wt_chi_plus: np.ndarray = None,
) -> AlanineScanResult:
    """Scan ``positions`` (author resnums), each mutated independently to
    alanine, and compare each mutant's chi_plus to the wild type's.
    Pass a precomputed ``wt_block_model``/``wt_chi_plus`` to skip
    recomputing the (unmutated) baseline."""
    if wt_chi_plus is None:
        cm_wt = compute_contact_map(structure)
        wt_block_model = build_blocks(ss_mask, cm_wt, block_size=block_size)
        wt_chi_plus = compute_coupling(structure, wt_block_model, ss_mask, params).chi_plus

    block_of_position = {
        int(resnum): int(wt_block_model.block_of_residue[i])
        for i, resnum in enumerate(structure.author_resnum)
    }

    result = AlanineScanResult(positions=list(positions), block_of_position=block_of_position, wt_chi_plus=wt_chi_plus)

    # Some blocks' chi_plus rows are entirely NaN (near-zero-probability
    # joint states, masked in compute_coupling as numerically unresolvable
    # rather than reported as noise) -- nanmean/nanstd over an all-NaN row
    # is expected here (result is legitimately NaN), just noisy to warn
    # about every time.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for resnum in positions:
            chi_plus_mut, bm_mut = mutant_chi_plus(structure, ss_mask, block_size, params, [resnum])
            assert bm_mut.nblocks == wt_block_model.nblocks, "mutation unexpectedly changed the block partition"
            ddg = chi_plus_mut - wt_chi_plus
            result.ddg_plus[resnum] = ddg
            result.mean_ddg_vector[resnum] = np.nanmean(ddg, axis=1)

        stacked = np.array([result.mean_ddg_vector[r] for r in positions])
        result.MR_mean = np.nanmean(stacked, axis=0)
        result.MR_std = np.nanstd(stacked, axis=0)

    return result
