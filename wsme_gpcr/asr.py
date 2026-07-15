"""Ancestral sequence reconstruction (ASR) integration.

Reads IQ-TREE2 `.state` per-site posterior probability files and uses
them to test whether a WSME fold-cooperativity result for a given
ancestral node is trustworthy, or an artifact of reconstruction
uncertainty at a small number of ambiguous sites -- rather than a real
property of the reconstructed protein.

An ancestral node's reconstructed sequence is a per-site point estimate
(the single most-probable residue at every alignment column), not a
real historical protein. AlphaFold will still confidently fold that
sequence regardless of whether it's a biologically coherent combination,
so a structure's pLDDT says nothing about reconstruction confidence. Two
independent gates are needed before treating a node's WSME result as
usable evolutionary signal:

  1. ``fold_ok`` -- does the node's own WSME landscape fold at all,
     ANYWHERE in a physically plausible xi range (``calibration.
     xi_fold_scan``'s ``folds_anywhere``)? **Not** a single-point test at
     one fixed reference xi -- a real paper reference receptor (4XNV/
     gpcr14i) was found to flip from 97.4% folded to 5.1% collapsed
     across a xi window under 0.7 J/mol (see FINDINGS.md's real-
     structure-control and xi-sweep entries), and testing every
     structure at one shared fixed xi turned out to misclassify most of
     a real 6-node ancestral test set purely by chance of which side of
     their own sharp transition that one point happened to fall on --
     every one of those six nodes, plus the real 4XNV control, in fact
     folds properly somewhere in the physical range.
  2. ``sensitivity_ok`` -- is that result robust to the node's own ASR
     reconstruction uncertainty? Tested by truncating every ambiguous
     site (IQ-TREE MAP posterior below a threshold) to alanine
     simultaneously and re-scanning: does the mutant still fold
     anywhere, and has its own transition point shifted substantially
     relative to the wild type's? This is NOT a true identity swap to
     the alternate reconstructed residue (that needs real rotamer/
     backbone modeling, i.e. actual re-folding, which this pipeline does
     not do) -- it tests whether the specific positions ASR is uncertain
     about are load-bearing for the fold outcome, a real, honest,
     partial robustness signal.

Both gates were developed against a real 6-node ancestral test case (see
FINDINGS.md's "Point-mutation sensitivity proxy" and "xi sweep" entries).
The original version of gate 1 tested fold quality at one fixed
reference xi and found 5 of 6 nodes "failed" -- the xi sweep then showed
every one of those six nodes (and a real paper reference receptor used
as a control) folds properly somewhere in the physical range, with a
single sharp transition whose location varies structure to structure.
Single-point testing was measuring which side of a shared, arbitrary
reference point a structure's own transition happened to fall on, not
whether it could fold -- gate 1 was corrected to use ``xi_fold_scan``
directly because of this finding.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .alanine_scan import EXCLUDED_FROM_SCAN, alanine_exclude_mask
from .blocking import BlockModel, build_blocks
from .calibration import (
    DEFAULT_XI_SCAN_RANGE_J_MOL,
    FOLD_WINDOW_FRAC,
    PAPER_XI_STD_J_MOL,
    XiFoldScanResult,
    xi_fold_scan,
)
from .contacts import compute_contact_map
from .pipeline import run_pipeline
from .structure import Structure
from .wsme import WSMEParams

DEFAULT_POSTERIOR_THRESHOLD = 0.8
# Coarser than the primary transition-mapping scan (0.1 J/mol) -- ~10x
# faster, still adequate to detect the kind of multi-J/mol shift that
# would indicate a real sensitivity, not sub-J/mol precision matching.
DEFAULT_SENSITIVITY_SCAN_STEP_J_MOL = 1.0
# How far a node's own transition point is allowed to move after
# truncating its ambiguous ASR positions before calling it NOT robust.
# Tied to the paper's own real inter-receptor xi variability (45 real
# GPCRs, std 2.76 J/mol) rather than an arbitrary pick: a shift bigger
# than natural WT-to-WT variation among real receptors is a real,
# structurally meaningful effect, not noise.
DEFAULT_TRANSITION_SHIFT_TOLERANCE_J_MOL = PAPER_XI_STD_J_MOL


@dataclass
class NodePosteriors:
    """Per-site IQ-TREE2 ASR posterior data for one internal tree node."""

    node: str
    site: np.ndarray            # (n_sites,) alignment site number, 1-based, file order
    state: list                 # (n_sites,) MAP state per site ("-" = gap/no data)
    map_posterior: np.ndarray   # (n_sites,) posterior probability of the MAP state
    second_state: list          # (n_sites,) second-most-likely state
    second_posterior: np.ndarray  # (n_sites,) its posterior probability


def parse_iqtree_state_file(path) -> dict:
    """Parse an IQ-TREE2 ``.state`` ancestral-state-reconstruction file
    (``Node\tSite\tState\tp_A\tp_R\t...`` columns, ``#``-prefixed header
    comments) into ``{node_name: NodePosteriors}``.

    Pure-Python TSV parsing, no pandas dependency -- these files are
    small (one row per node per alignment site) and simply structured.
    """
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        rows_by_node = {}
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
                continue
            rows_by_node.setdefault(row[0], []).append(row)

    if header is None:
        raise ValueError(f"{path}: no header row found (expected a line starting with 'Node\tSite\tState...')")

    aa_cols = [c for c in header if c.startswith("p_")]
    if not aa_cols:
        raise ValueError(f"{path}: no p_<AA> posterior columns found in header {header!r}")
    aa_names = [c[2:] for c in aa_cols]
    aa_col_index = [header.index(c) for c in aa_cols]
    site_idx, state_idx = header.index("Site"), header.index("State")

    result = {}
    for node, rows in rows_by_node.items():
        rows = sorted(rows, key=lambda r: int(r[site_idx]))
        site = np.array([int(r[site_idx]) for r in rows])
        state = [r[state_idx] for r in rows]
        posteriors = np.array([[float(r[i]) for i in aa_col_index] for r in rows])

        n = len(rows)
        map_idx = posteriors.argmax(axis=1)
        map_posterior = posteriors[np.arange(n), map_idx]
        masked = posteriors.copy()
        masked[np.arange(n), map_idx] = -1.0  # exclude the MAP state, find the runner-up
        second_idx = masked.argmax(axis=1)
        second_posterior = masked[np.arange(n), second_idx]
        second_state = [aa_names[i] for i in second_idx]

        result[node] = NodePosteriors(
            node=node, site=site, state=state, map_posterior=map_posterior,
            second_state=second_state, second_posterior=second_posterior,
        )
    return result


def site_to_resnum(node_posteriors: NodePosteriors) -> np.ndarray:
    """Map each alignment site to a 1-based residue number, assuming
    this node's own FULL (untruncated) reconstructed sequence starts at
    resnum 1 -- i.e. resnum = the cumulative count of non-gap sites up
    to and including this one. Only meaningful at non-gap sites (check
    ``node_posteriors.state[i] != "-"``); gap sites get the previous
    residue's resnum repeated, which is not a valid resnum for that row.
    """
    non_gap = np.array([s != "-" for s in node_posteriors.state])
    return np.cumsum(non_gap)


def ambiguous_core_resnums(node_posteriors: NodePosteriors, structure: Structure,
                            posterior_threshold: float = DEFAULT_POSTERIOR_THRESHOLD) -> list:
    """Author resnums, restricted to ``structure``'s own resolved
    residue range, whose IQ-TREE MAP posterior is below
    ``posterior_threshold`` -- the positions eligible for
    ``run_asr_sensitivity_check``.

    Excludes gap sites, sites outside the structure's own resnum range
    (e.g. a truncated ordered-core structure -- see the disorder-scope
    diagnosis in FINDINGS.md for why truncation is often the right thing
    to have done upstream), and ALA/GLY/PRO (no meaningful alanine
    mutation there, matching ``alanine_scan.EXCLUDED_FROM_SCAN``).
    """
    resnum = site_to_resnum(node_posteriors)
    non_gap = np.array([s != "-" for s in node_posteriors.state])
    core_resnums = set(int(x) for x in structure.author_resnum)
    resname_by_resnum = {int(rn): rname for rn, rname in zip(structure.author_resnum, structure.resname)}

    out = []
    for i in range(len(node_posteriors.site)):
        if not non_gap[i]:
            continue
        rn = int(resnum[i])
        if rn not in core_resnums:
            continue
        if node_posteriors.map_posterior[i] >= posterior_threshold:
            continue
        if resname_by_resnum.get(rn) in EXCLUDED_FROM_SCAN:
            continue
        out.append(rn)
    return out


def _transition_midpoint(scan: XiFoldScanResult) -> float:
    """Midpoint xi of a scan's single fold/collapse transition, or
    ``None`` if the scan shows zero transitions (folds -- or doesn't --
    consistently across the whole scanned range, no sharp switch to
    locate) or more than one (not a clean single switch; the scanned
    range may need widening/narrowing for that structure)."""
    if scan.n_transitions != 1:
        return None
    fold_ok_mask = scan.fold_fracs >= (1.0 - FOLD_WINDOW_FRAC)
    for i in range(1, len(fold_ok_mask)):
        if fold_ok_mask[i] != fold_ok_mask[i - 1]:
            return float((scan.xi_values_j_mol[i - 1] + scan.xi_values_j_mol[i]) / 2.0)
    return None


@dataclass
class AsrSensitivityResult:
    node: str
    posterior_threshold: float
    nblocks: int
    ambiguous_resnums: list = field(default_factory=list)
    wt_folds_anywhere: bool = False
    wt_transition_xi_j_mol: float = None
    wt_scan: XiFoldScanResult = None
    mutant_folds_anywhere: bool = False
    mutant_transition_xi_j_mol: float = None
    mutant_scan: XiFoldScanResult = None
    transition_shift_j_mol: float = None   # mutant - WT transition location (signed); None if not comparable
    fold_ok: bool = False        # wt_folds_anywhere -- does the WT structure fold ANYWHERE plausible?
    sensitivity_ok: bool = False  # is that result robust to this node's own ASR ambiguity?
    trustworthy: bool = False    # fold_ok AND sensitivity_ok

    def reason(self) -> str:
        """One-line, human-readable classification reason."""
        if self.trustworthy:
            return "trustworthy: folds properly somewhere plausible and robust to its own ASR ambiguity"
        if not self.fold_ok:
            return "not trustworthy: does not fold anywhere in the scanned xi range"
        if not self.mutant_folds_anywhere:
            return (f"not trustworthy: folds at wild type, but truncating its {len(self.ambiguous_resnums)} "
                    "ambiguous position(s) abolishes folding entirely -- likely a reconstruction-uncertainty "
                    "artifact, not real biology")
        shift_desc = f"{self.transition_shift_j_mol:+.1f} J/mol" if self.transition_shift_j_mol is not None else "not comparable"
        return (f"not trustworthy: folds, but its transition point is NOT robust to its own ASR ambiguity "
                f"(shift={shift_desc} on {len(self.ambiguous_resnums)} ambiguous position(s)) -- likely a "
                "reconstruction-uncertainty artifact, not real biology")


def run_asr_sensitivity_check(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray,
                               params: WSMEParams, ambiguous_resnums: list,
                               xi_range_j_mol: tuple = DEFAULT_XI_SCAN_RANGE_J_MOL,
                               wt_step_j_mol: float = None,
                               mutant_step_j_mol: float = DEFAULT_SENSITIVITY_SCAN_STEP_J_MOL,
                               shift_tolerance_j_mol: float = DEFAULT_TRANSITION_SHIFT_TOLERANCE_J_MOL,
                               wt_scan: XiFoldScanResult = None,
                               node: str = "", posterior_threshold: float = float("nan")) -> AsrSensitivityResult:
    """Does this structure fold anywhere in a plausible xi range
    (``fold_ok``), and is that robust to truncating its own ambiguous ASR
    positions to alanine simultaneously (``sensitivity_ok``)? Robustness
    is judged two ways: does the mutant still fold anywhere at all, and
    -- if both wild type and mutant have one clean, locatable transition
    -- has that transition point shifted by more than
    ``shift_tolerance_j_mol`` (default: the paper's own real
    inter-receptor xi standard deviation, 2.76 J/mol). If one of the two
    has a clean single transition and the other doesn't (e.g. mutant
    always folds/never folds across the whole scan while WT had a sharp
    switch), that qualitative change alone is treated as not robust.

    Pass a precomputed ``wt_scan`` (e.g. already computed at finer
    resolution for other purposes) to skip re-scanning the wild type --
    otherwise it's computed fresh at ``wt_step_j_mol`` (default: falls
    back to ``calibration.DEFAULT_XI_SCAN_STEP_J_MOL``, the same fine
    default ``xi_fold_scan`` itself uses). The mutant is always scanned
    fresh, at the coarser ``mutant_step_j_mol`` by default -- sufficient
    to detect a multi-J/mol shift without the cost of matching the WT
    scan's full resolution.

    ``node``/``posterior_threshold`` are optional descriptive metadata
    copied onto the result, not used in the computation itself -- pass
    them when calling this directly (rather than via
    ``evaluate_node_trustworthiness``) if you want a self-describing
    result.
    """
    if wt_scan is None:
        wt_scan = xi_fold_scan(structure, block_model, ss_mask, params, xi_range_j_mol=xi_range_j_mol,
                                step_j_mol=wt_step_j_mol if wt_step_j_mol is not None else 0.1)
    wt_transition = _transition_midpoint(wt_scan)
    fold_ok = wt_scan.folds_anywhere

    if not ambiguous_resnums:
        return AsrSensitivityResult(
            node=node, posterior_threshold=posterior_threshold, nblocks=block_model.nblocks,
            ambiguous_resnums=[], wt_folds_anywhere=wt_scan.folds_anywhere,
            wt_transition_xi_j_mol=wt_transition, wt_scan=wt_scan,
            mutant_folds_anywhere=wt_scan.folds_anywhere, mutant_transition_xi_j_mol=wt_transition,
            mutant_scan=wt_scan, transition_shift_j_mol=0.0,
            fold_ok=fold_ok, sensitivity_ok=True, trustworthy=fold_ok,
        )

    exclude = alanine_exclude_mask(structure, ambiguous_resnums)
    cm_mut = compute_contact_map(structure, exclude_atoms=exclude)
    bm_mut = build_blocks(ss_mask, cm_mut, block_size=block_model.block_size)
    if bm_mut.nblocks != block_model.nblocks:
        raise RuntimeError(
            "mutation unexpectedly changed the block partition "
            f"({bm_mut.nblocks} vs {block_model.nblocks}) -- block boundaries should depend only on "
            "ss_mask, which this check does not modify; this indicates a real bug, not expected behavior"
        )

    mutant_scan = xi_fold_scan(structure, bm_mut, ss_mask, params, xi_range_j_mol=xi_range_j_mol,
                               step_j_mol=mutant_step_j_mol)
    mutant_transition = _transition_midpoint(mutant_scan)

    if wt_transition is not None and mutant_transition is not None:
        shift = mutant_transition - wt_transition
        sensitivity_ok = mutant_scan.folds_anywhere and abs(shift) <= shift_tolerance_j_mol
    elif wt_transition is None and mutant_transition is None:
        # neither has an isolated transition in-range (e.g. both fold
        # consistently across the whole scan) -- no shift to measure,
        # but that's itself a robust (or robustly absent) outcome.
        shift = None
        sensitivity_ok = mutant_scan.folds_anywhere == wt_scan.folds_anywhere
    else:
        # one has a clean single transition, the other doesn't -- a real
        # qualitative change, not treated as robust.
        shift = None
        sensitivity_ok = False

    trustworthy = fold_ok and sensitivity_ok

    return AsrSensitivityResult(
        node=node, posterior_threshold=posterior_threshold, nblocks=block_model.nblocks,
        ambiguous_resnums=list(ambiguous_resnums), wt_folds_anywhere=wt_scan.folds_anywhere,
        wt_transition_xi_j_mol=wt_transition, wt_scan=wt_scan,
        mutant_folds_anywhere=mutant_scan.folds_anywhere, mutant_transition_xi_j_mol=mutant_transition,
        mutant_scan=mutant_scan, transition_shift_j_mol=shift,
        fold_ok=fold_ok, sensitivity_ok=sensitivity_ok, trustworthy=trustworthy,
    )


def evaluate_node_trustworthiness(pdb_path, node_posteriors: NodePosteriors, params: WSMEParams = None,
                                   use_dssp: bool = True,
                                   posterior_threshold: float = DEFAULT_POSTERIOR_THRESHOLD,
                                   xi_range_j_mol: tuple = DEFAULT_XI_SCAN_RANGE_J_MOL,
                                   wt_step_j_mol: float = None,
                                   mutant_step_j_mol: float = DEFAULT_SENSITIVITY_SCAN_STEP_J_MOL,
                                   shift_tolerance_j_mol: float = DEFAULT_TRANSITION_SHIFT_TOLERANCE_J_MOL,
                                   wt_scan: XiFoldScanResult = None) -> AsrSensitivityResult:
    """End-to-end single-call evaluation of one ancestral node: load and
    fold ``pdb_path`` (via ``run_pipeline``), identify this node's own
    ambiguous ASR sites within the resolved structure, and run the
    sensitivity check. ``use_dssp=True`` by default -- see FINDINGS.md's
    block-partition audit for why real DSSP blocking matters here.
    """
    if params is None:
        params = WSMEParams()
    r = run_pipeline(pdb_path, ph=7.0, use_dssp=use_dssp, params=params)
    ambiguous = ambiguous_core_resnums(node_posteriors, r.structure, posterior_threshold)
    return run_asr_sensitivity_check(
        r.structure, r.block_model, r.ss_mask, params, ambiguous,
        xi_range_j_mol=xi_range_j_mol, wt_step_j_mol=wt_step_j_mol, mutant_step_j_mol=mutant_step_j_mol,
        shift_tolerance_j_mol=shift_tolerance_j_mol, wt_scan=wt_scan,
        node=node_posteriors.node, posterior_threshold=posterior_threshold,
    )
