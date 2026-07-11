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
re-alignment needed. That also means every mutant shares the wild type's
segment-pair combinatorics (see wsme._build_pair_indices), which this
module builds once per scan and reuses, instead of re-deriving it for
every mutant.

This module is written to be applied to *any* structure and *any* set of
positions -- not tied to a specific receptor. ``run_alanine_scan`` scans
every eligible residue by default (matching the paper's "large-scale"
approach); pass ``positions`` or ``max_positions`` for a faster, smaller
run. Each mutant costs roughly the same as one coupling-matrix
computation (seconds, scaling with structure size), so a full ~250-300
site GPCR scan is a tens-of-minutes job -- see ``estimate_scan_seconds``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .blocking import BlockModel, build_blocks
from .contacts import compute_contact_map
from .coupling import compute_coupling
from .structure import Structure
from .wsme import WSMEParams, _build_pair_indices

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


def subsample_positions(positions: list, max_positions: int) -> list:
    """Evenly-spaced subsample across the sequence (rather than just the
    first N), so a capped scan still covers the whole receptor instead of
    only its N-terminal half."""
    positions = list(positions)
    if max_positions is None or max_positions >= len(positions):
        return positions
    idx = np.linspace(0, len(positions) - 1, max_positions).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [positions[i] for i in idx]


def estimate_scan_seconds(n_positions: int, seconds_per_position: float = 8.0) -> float:
    """Rough wall-clock estimate for a scan of this many positions (plus
    one wild-type baseline). ``seconds_per_position`` should be calibrated
    to your structure's size -- larger receptors (more blocks) cost more
    per mutant; time a single mutant first for a size you haven't
    scanned before."""
    return (n_positions + 1) * seconds_per_position


def _ca_position(structure: Structure, resnum: int) -> np.ndarray:
    ridx = int(np.where(structure.author_resnum == resnum)[0][0])
    mask = (structure.atom_resindex == ridx) & (np.array(structure.atom_name) == "CA")
    if not np.any(mask):
        return None
    return structure.coord[mask][0]


def block_ca_centroids(structure: Structure, block_model: BlockModel) -> np.ndarray:
    """(nblocks, 3) mean CA position of each block's constituent residues."""
    ca_mask = np.array(structure.atom_name) == "CA"
    ca_coord = structure.coord[ca_mask]
    ca_resindex = structure.atom_resindex[ca_mask]
    block_of_res = block_model.block_of_residue
    nb = block_model.nblocks
    centroids = np.full((nb, 3), np.nan)
    for b in range(nb):
        residues = np.where(block_of_res == b)[0]
        atoms = np.isin(ca_resindex, residues)
        if np.any(atoms):
            centroids[b] = ca_coord[atoms].mean(axis=0)
    return centroids


def mutant_chi_plus(structure: Structure, ss_mask: np.ndarray, block_size: int, params: WSMEParams, resnums,
                     pair_indices=None) -> tuple:
    """chi_plus (positive coupling free energy matrix) for the structure
    with the given residue(s) computationally mutated to alanine."""
    exclude = alanine_exclude_mask(structure, resnums)
    cm = compute_contact_map(structure, exclude_atoms=exclude)
    bm = build_blocks(ss_mask, cm, block_size=block_size)
    coupling = compute_coupling(structure, bm, ss_mask, params, pair_indices=pair_indices)
    return coupling.chi_plus, bm


@dataclass
class AlanineScanResult:
    positions: list  # author resnums scanned
    block_of_position: dict  # resnum -> block index
    block_ca_centroid: np.ndarray  # (nblocks, 3)
    wt_chi_plus: np.ndarray  # (nblocks, nblocks)
    ddg_plus: dict = field(default_factory=dict)  # resnum -> (nblocks, nblocks) chi_plus_mut - chi_plus_wt
    mean_ddg_vector: dict = field(default_factory=dict)  # resnum -> (nblocks,) row-averaged DeltaDeltaG+
    MR_mean: np.ndarray = None  # (nblocks,) mean mutational response across all scanned positions
    MR_std: np.ndarray = None  # (nblocks,) std of mutational response

    def top_hits(self, n: int = 10) -> list:
        """Mutation sites ranked by total perturbation magnitude
        (sum of |mean_ddg_vector|), largest first -- the paper's approach
        to picking out sites like K296/N302/M317 for closer inspection."""
        scores = [
            (resnum, float(np.nansum(np.abs(v))))
            for resnum, v in self.mean_ddg_vector.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def ddg_vs_distance(self, resnum: int):
        """(distances, ddg_values) for one scanned mutation: per-block
        DeltaDeltaG+ vs. CA-CA distance (Angstrom) from the mutated
        residue's own block centroid -- the paper's Fig. 7d/f/h."""
        v = self.mean_ddg_vector[resnum]
        b = self.block_of_position[resnum]
        origin = self.block_ca_centroid[b]
        dist = np.linalg.norm(self.block_ca_centroid - origin, axis=1)
        return dist, v

    def to_records(self) -> list:
        """One row per (mutation site, block) pair -- flat table for CSV export."""
        rows = []
        for resnum in self.positions:
            v = self.mean_ddg_vector[resnum]
            for b, val in enumerate(v):
                rows.append({"mutated_resnum": resnum, "block": b, "mean_ddG+": float(val)})
        return rows


def run_alanine_scan(
    structure: Structure,
    ss_mask: np.ndarray,
    params: WSMEParams,
    positions=None,
    max_positions: int = None,
    block_size: int = 4,
    wt_block_model: BlockModel = None,
    wt_chi_plus: np.ndarray = None,
    progress_callback=None,
) -> AlanineScanResult:
    """Scan ``positions`` (author resnums), each mutated independently to
    alanine, and compare each mutant's chi_plus to the wild type's.

    ``positions=None`` scans every eligible residue (``scannable_positions``)
    -- a full receptor-wide scan. Pass an explicit list to target specific
    sites, or ``max_positions`` to evenly subsample the full site list for
    a faster, still whole-structure-covering run.

    Pass a precomputed ``wt_block_model``/``wt_chi_plus`` to skip
    recomputing the (unmutated) baseline. ``progress_callback(resnum, i,
    total, elapsed_seconds)`` is called after each mutant, if given.
    """
    if positions is None:
        positions = scannable_positions(structure)
    if max_positions is not None:
        positions = subsample_positions(positions, max_positions)

    if wt_chi_plus is None:
        cm_wt = compute_contact_map(structure)
        wt_block_model = build_blocks(ss_mask, cm_wt, block_size=block_size)
        pair_indices = _build_pair_indices(wt_block_model.nblocks)
        wt_chi_plus = compute_coupling(structure, wt_block_model, ss_mask, params, pair_indices=pair_indices).chi_plus
    else:
        pair_indices = _build_pair_indices(wt_block_model.nblocks)

    block_of_position = {
        int(resnum): int(wt_block_model.block_of_residue[i])
        for i, resnum in enumerate(structure.author_resnum)
    }
    block_ca_centroid = block_ca_centroids(structure, wt_block_model)

    result = AlanineScanResult(
        positions=list(positions),
        block_of_position=block_of_position,
        block_ca_centroid=block_ca_centroid,
        wt_chi_plus=wt_chi_plus,
    )

    # Some blocks' chi_plus rows are entirely NaN (near-zero-probability
    # joint states, masked in compute_coupling as numerically unresolvable
    # rather than reported as noise) -- nanmean/nanstd over an all-NaN row
    # is expected here (result is legitimately NaN), just noisy to warn
    # about every time.
    t0 = time.time()
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, resnum in enumerate(positions):
            chi_plus_mut, bm_mut = mutant_chi_plus(
                structure, ss_mask, block_size, params, [resnum], pair_indices=pair_indices,
            )
            assert bm_mut.nblocks == wt_block_model.nblocks, "mutation unexpectedly changed the block partition"
            ddg = chi_plus_mut - wt_chi_plus
            result.ddg_plus[resnum] = ddg
            result.mean_ddg_vector[resnum] = np.nanmean(ddg, axis=1)
            if progress_callback:
                progress_callback(resnum, i, len(positions), time.time() - t0)

        stacked = np.array([result.mean_ddg_vector[r] for r in positions])
        result.MR_mean = np.nanmean(stacked, axis=0)
        result.MR_std = np.nanstd(stacked, axis=0)

    return result


def ph_sensitivity_table(scan_by_ph: dict, n: int = 15) -> list:
    """Rank mutation sites by how much their perturbation magnitude
    (sum |mean_ddg_vector|) swings across the scanned pH values.

    ``scan_by_ph`` maps pH -> AlanineScanResult (independent scans at each
    pH, from ``run_alanine_scan_pipeline_multi_ph``). Alanine-scan results
    are themselves pH-dependent -- pH changes which atoms carry a
    titratable charge, which feeds back into the contact map and hence the
    coupling matrix each mutant is compared against -- so a mutation's
    apparent structural importance can shift with pH. A large swing here
    flags a site whose *coupling role*, not just its own charge state,
    looks pH-modulated: a candidate conformational pH sensor, distinct
    from (but complementary to) the buried-ionizable-network view in
    ``ionizable_network``.

    Only sites appearing in at least one pH's ``top_hits(n)`` are
    included (comparing every scanned site at every pH is usually just
    noise for sites with a near-zero effect everywhere).
    """
    ph_values = sorted(scan_by_ph.keys())
    resnums = set()
    for scan in scan_by_ph.values():
        resnums.update(int(r) for r, _ in scan.top_hits(n))

    rows = []
    for resnum in sorted(resnums):
        scores = {}
        for ph in ph_values:
            v = scan_by_ph[ph].mean_ddg_vector.get(resnum)
            scores[ph] = float(np.nansum(np.abs(v))) if v is not None else float("nan")
        finite = [s for s in scores.values() if np.isfinite(s)]
        spread = (max(finite) - min(finite)) if finite else float("nan")
        rows.append({"resnum": resnum, "scores_by_ph": scores, "ph_spread": spread})

    rows.sort(key=lambda r: r["ph_spread"] if np.isfinite(r["ph_spread"]) else -1.0, reverse=True)
    return rows


def residue_ph_features(scan_by_ph: dict) -> tuple:
    """Build a per-residue feature matrix from a multi-pH alanine scan,
    for PCA/clustering: each row is one scanned residue's
    ``mean_ddg_vector`` at every pH, concatenated -- so it captures both
    *which blocks* a mutation perturbs and *how that pattern shifts
    across pH*, not just an overall magnitude.

    ``scan_by_ph`` maps pH -> AlanineScanResult, all sharing the same
    scanned positions (as produced by ``run_alanine_scan_pipeline_multi_ph``).

    Returns ``(resnums, features, magnitude, ph_spread)``:
      - ``features``: (n_residues, nblocks * n_ph) array. NaN entries
        (numerically unresolvable coupling -- see ``compute_coupling`` --
        not missing data) are treated as zero perturbation so PCA and
        clustering stay well-defined.
      - ``magnitude``: (n_residues,) mean total perturbation
        (sum |mean_ddg_vector|) across pH -- "does this mutation matter
        at all," independent of pH.
      - ``ph_spread``: (n_residues,) max-min of that per-pH total
        across pH -- "does how much it matters depend on pH."
    """
    ph_values = sorted(scan_by_ph.keys())
    positions = list(scan_by_ph[ph_values[0]].positions)

    rows, magnitude, ph_spread = [], [], []
    for resnum in positions:
        per_ph_vectors, per_ph_scores = [], []
        for ph in ph_values:
            v = np.nan_to_num(scan_by_ph[ph].mean_ddg_vector[resnum], nan=0.0)
            per_ph_vectors.append(v)
            per_ph_scores.append(float(np.sum(np.abs(v))))
        rows.append(np.concatenate(per_ph_vectors))
        magnitude.append(float(np.mean(per_ph_scores)))
        ph_spread.append(float(max(per_ph_scores) - min(per_ph_scores)))

    return np.array(positions), np.array(rows), np.array(magnitude), np.array(ph_spread)


def pca_cluster_residues(features: np.ndarray, n_components: int = 2, n_clusters: int = 4, seed: int = 0) -> tuple:
    """PCA (via SVD; no extra ML dependency beyond scipy, already required)
    plus k-means clustering (``scipy.cluster.vq``) on a per-residue feature
    matrix from ``residue_ph_features`` -- groups mutation sites by the
    *shape* of their per-block x per-pH coupling perturbation pattern.

    Returns ``(coords, labels, explained_variance_ratio)``: ``coords`` is
    (n_residues, n_components), ``labels`` is (n_residues,) integer cluster
    IDs, ``explained_variance_ratio`` is (n_components,).
    """
    from scipy.cluster.vq import kmeans2

    X = features - features.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # guard columns with zero variance (e.g. an always-NaN block)
    Xn = X / std

    U, S, _ = np.linalg.svd(Xn, full_matrices=False)
    n_components = min(n_components, U.shape[1])
    coords = U[:, :n_components] * S[:n_components]
    explained_variance_ratio = (S ** 2 / np.sum(S ** 2))[:n_components]

    k = min(n_clusters, coords.shape[0])
    _, labels = kmeans2(coords, k, minit="++", seed=seed)

    return coords, labels, explained_variance_ratio


def ph_cluster_table(scan_by_ph: dict, n_clusters: int = 4, seed: int = 0) -> list:
    """One row per scanned residue: PCA coordinates, cluster assignment,
    overall stability-effect magnitude, and pH-dependence -- the full
    per-residue table backing ``plotting.plot_alanine_ph_pca`` /
    ``plot_alanine_ph_magnitude_vs_sensitivity``, exportable as CSV."""
    resnums, features, magnitude, ph_spread = residue_ph_features(scan_by_ph)
    coords, labels, evr = pca_cluster_residues(features, n_clusters=n_clusters, seed=seed)
    rows = [
        {
            "resnum": int(resnums[i]),
            "cluster": int(labels[i]),
            "magnitude": float(magnitude[i]),
            "ph_spread": float(ph_spread[i]),
            "pc1": float(coords[i, 0]),
            "pc2": float(coords[i, 1]) if coords.shape[1] > 1 else float("nan"),
        }
        for i in range(len(resnums))
    ]
    rows.sort(key=lambda r: r["magnitude"], reverse=True)
    return rows
