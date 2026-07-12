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

  1. ``fold_ok`` -- does the node's own WSME landscape find its
     reference fold within the top ``calibration.FOLD_WINDOW_FRAC`` of
     the reaction coordinate at all?
  2. ``sensitivity_ok`` -- is that result robust to the node's own ASR
     reconstruction uncertainty? Tested by truncating every ambiguous
     site (IQ-TREE MAP posterior below a threshold) to alanine
     simultaneously and checking whether the fold fraction survives.
     This is NOT a true identity swap to the alternate reconstructed
     residue (that needs real rotamer/backbone modeling, i.e. actual
     re-folding, which this pipeline does not do) -- it tests whether
     the specific positions ASR is uncertain about are load-bearing for
     the fold outcome, a real, honest, partial robustness signal.

Both gates were validated against a real 4-node ancestral test case
(see FINDINGS.md's "Point-mutation sensitivity proxy" entry): node_148
folded well (90.4%) and was completely insensitive to its own 44
ambiguous positions (0.0pp delta) -- a trustworthy result. node_20 had
the *highest* average ASR confidence of the four but collapsed by 43.7
percentage points when just its 16 truly ambiguous positions were
truncated -- direct evidence its own fold result (59.2%, already below
the fold_ok bar) is substantially an ASR-uncertainty artifact, not a
stable property of the ancestral sequence. Average per-site confidence
alone did not predict this; whether the *few* uncertain positions happen
to be load-bearing does.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from .alanine_scan import EXCLUDED_FROM_SCAN, alanine_exclude_mask
from .blocking import BlockModel, build_blocks
from .calibration import FOLD_WINDOW_FRAC
from .contacts import compute_contact_map
from .pipeline import run_pipeline
from .structure import Structure
from .wsme import WSMEParams, run_wsme

DEFAULT_POSTERIOR_THRESHOLD = 0.8
# 10 percentage points of fold fraction -- a documented, defensible
# default, not independently re-derived from a larger benchmark. Matches
# the real gap observed in the validation run between a robust result
# (node_148, 0.0pp) and an artifact-driven one (node_20, 43.7pp on just
# 16 positions) -- see this module's docstring.
DEFAULT_DELTA_TOLERANCE_FRAC = 0.10


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


@dataclass
class AsrSensitivityResult:
    node: str
    posterior_threshold: float
    nblocks: int
    ambiguous_resnums: list = field(default_factory=list)
    wt_fold_frac: float = 0.0
    mutant_fold_frac: float = 0.0
    delta_frac: float = 0.0
    fold_ok: bool = False        # WT landscape's own fold-quality gate (FOLD_WINDOW_FRAC)
    sensitivity_ok: bool = False  # is fold_ok robust to this node's own ASR ambiguity?
    trustworthy: bool = False    # fold_ok AND sensitivity_ok

    def reason(self) -> str:
        """One-line, human-readable classification reason."""
        if self.trustworthy:
            return "trustworthy: folds properly and robust to its own ASR ambiguity"
        if not self.fold_ok:
            return f"not trustworthy: does not fold (min at {self.wt_fold_frac:.1%} of reaction coordinate)"
        return (f"not trustworthy: folds ({self.wt_fold_frac:.1%}) but is NOT robust to its own ASR "
                f"ambiguity (delta={self.delta_frac:+.1%} on {len(self.ambiguous_resnums)} "
                f"ambiguous position(s)) -- likely a reconstruction-uncertainty artifact, not real biology")


def run_asr_sensitivity_check(structure: Structure, block_model: BlockModel, ss_mask: np.ndarray,
                               params: WSMEParams, ambiguous_resnums: list,
                               delta_tolerance_frac: float = DEFAULT_DELTA_TOLERANCE_FRAC,
                               node: str = "", posterior_threshold: float = float("nan")) -> AsrSensitivityResult:
    """Truncate every resnum in ``ambiguous_resnums`` to alanine
    simultaneously (same backbone, same block partition -- block
    boundaries depend only on ``ss_mask``, not on which side-chain atoms
    are excluded from the contact map) and compare the resulting WSME
    1D landscape's fold fraction to the unperturbed baseline.

    ``node``/``posterior_threshold`` are optional descriptive metadata
    copied onto the result, not used in the computation itself -- pass
    them when calling this directly (rather than via
    ``evaluate_node_trustworthiness``) if you want a self-describing
    result.
    """
    res_wt = run_wsme(structure, block_model, ss_mask, params)
    amin_wt = int(np.argmin(res_wt.fes))
    wt_frac = float(res_wt.n_values[amin_wt] / block_model.nblocks)
    fold_ok = wt_frac >= (1.0 - FOLD_WINDOW_FRAC)

    if not ambiguous_resnums:
        return AsrSensitivityResult(
            node=node, posterior_threshold=posterior_threshold, nblocks=block_model.nblocks,
            ambiguous_resnums=[], wt_fold_frac=wt_frac, mutant_fold_frac=wt_frac, delta_frac=0.0,
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

    res_mut = run_wsme(structure, bm_mut, ss_mask, params)
    amin_mut = int(np.argmin(res_mut.fes))
    mut_frac = float(res_mut.n_values[amin_mut] / bm_mut.nblocks)

    delta = mut_frac - wt_frac
    sensitivity_ok = abs(delta) <= delta_tolerance_frac
    trustworthy = fold_ok and sensitivity_ok

    return AsrSensitivityResult(
        node=node, posterior_threshold=posterior_threshold, nblocks=block_model.nblocks,
        ambiguous_resnums=list(ambiguous_resnums), wt_fold_frac=wt_frac, mutant_fold_frac=mut_frac,
        delta_frac=delta, fold_ok=fold_ok, sensitivity_ok=sensitivity_ok, trustworthy=trustworthy,
    )


def evaluate_node_trustworthiness(pdb_path, node_posteriors: NodePosteriors, params: WSMEParams = None,
                                   use_dssp: bool = True,
                                   posterior_threshold: float = DEFAULT_POSTERIOR_THRESHOLD,
                                   delta_tolerance_frac: float = DEFAULT_DELTA_TOLERANCE_FRAC) -> AsrSensitivityResult:
    """End-to-end single-call evaluation of one ancestral node: load and
    fold ``pdb_path`` (via ``run_pipeline``), identify this node's own
    ambiguous ASR sites within the resolved structure, and run the
    sensitivity check. ``use_dssp=True`` by default -- see FINDINGS.md's
    block-partition audit for why real DSSP blocking matters here (the
    geometric heuristic gave a materially different, less trustworthy
    fold-quality picture for this exact 4-node test case).
    """
    if params is None:
        params = WSMEParams()
    r = run_pipeline(pdb_path, ph=7.0, use_dssp=use_dssp, params=params)
    ambiguous = ambiguous_core_resnums(node_posteriors, r.structure, posterior_threshold)
    return run_asr_sensitivity_check(
        r.structure, r.block_model, r.ss_mask, params, ambiguous,
        delta_tolerance_frac=delta_tolerance_frac, node=node_posteriors.node,
        posterior_threshold=posterior_threshold,
    )
