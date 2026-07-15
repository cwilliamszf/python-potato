"""Structural clustering of a predicted-model ensemble, to pick a diverse, non-redundant
set of representative conformations out of (potentially hundreds of) ColabFold outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Minimum RMSD between two (N, 3) coordinate sets after optimal rigid superposition."""
    a = coords_a - coords_a.mean(axis=0)
    b = coords_b - coords_b.mean(axis=0)
    cov = a.T @ b
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    corr = np.diag([1.0, 1.0, d])
    rot = vt.T @ corr @ u.T
    a_rot = a @ rot.T
    diff = a_rot - b
    return float(np.sqrt((diff**2).sum(axis=1).mean()))


def pairwise_rmsd_matrix(coord_sets: list[np.ndarray]) -> np.ndarray:
    n = len(coord_sets)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = kabsch_rmsd(coord_sets[i], coord_sets[j])
            mat[i, j] = mat[j, i] = d
    return mat


def cluster_by_rmsd(rmsd_matrix: np.ndarray, distance_cutoff: float) -> np.ndarray:
    """Average-linkage clustering on an RMSD distance matrix. Returns a 1-indexed label
    array of length n. `distance_cutoff` is in the same units as the RMSD matrix (Angstrom)
    and controls how structurally distinct two models must be to land in separate clusters.
    """
    condensed = squareform(rmsd_matrix, checks=False)
    z = linkage(condensed, method="average")
    return fcluster(z, t=distance_cutoff, criterion="distance")


def select_representatives(labels: np.ndarray, quality_scores: list[float]) -> dict[int, int]:
    """For each cluster label, pick the member index with the highest quality score
    (e.g. mean pLDDT) as the cluster representative."""
    best: dict[int, tuple[int, float]] = {}
    for idx, (lab, score) in enumerate(zip(labels, quality_scores)):
        if lab not in best or score > best[lab][1]:
            best[lab] = (idx, score)
    return {lab: idx for lab, (idx, _score) in best.items()}


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical (Torgerson) MDS embedding of a distance matrix, for 2D visualization of
    where each model sits in conformational space relative to the others."""
    d2 = distance_matrix**2
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:n_components]
    eigvecs = eigvecs[:, order][:, :n_components]
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
    return eigvecs * np.sqrt(eigvals_clipped)


def load_ca_coords(pdb_path: str | Path, chain: str = "A") -> dict[int, np.ndarray]:
    """Extract {residue_number: CA coordinate} from a PDB file for one chain."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("m", str(pdb_path))
    coords: dict[int, np.ndarray] = {}
    for model in structure:
        for ch in model:
            if ch.id != chain:
                continue
            for residue in ch:
                if "CA" in residue:
                    coords[residue.id[1]] = np.array(residue["CA"].coord, dtype=float)
        break
    return coords


def common_core_coords(
    coord_dicts: list[dict[int, np.ndarray]], residue_range: tuple[int, int] | None = None
) -> list[np.ndarray]:
    """Reduce a list of {resnum: coord} dicts to aligned (N, 3) arrays over the residue
    numbers shared by every model (optionally restricted to `residue_range`, e.g. the
    transmembrane core, to avoid flexible loops/termini dominating the RMSD)."""
    common = set.intersection(*(set(d.keys()) for d in coord_dicts))
    if residue_range is not None:
        lo, hi = residue_range
        common = {r for r in common if lo <= r <= hi}
    ordered = sorted(common)
    if len(ordered) < 3:
        raise ValueError("Fewer than 3 common residues across models; check inputs/range")
    return [np.array([d[r] for r in ordered]) for d in coord_dicts]
