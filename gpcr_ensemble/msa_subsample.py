"""Generate a diverse set of input MSAs from one full a3m to drive AlphaFold2/ColabFold
toward different conformational basins.

Two complementary strategies from the literature, combined:

- Random shallow subsampling (Del Alamo et al. 2022, eLife): reducing MSA depth removes
  the co-evolutionary signal that locks AF2 onto a single dominant state, letting the
  structure module explore alternate low-energy conformations. Run many random subsets
  at several depths/fractions.
- Sequence-cluster subsampling (AF-Cluster; Wayment-Steele et al. 2024, Nature): splitting
  the MSA into clusters of similar sequences (e.g. paralog/ortholog sub-families) and
  folding each cluster's shallow MSA separately biases each run toward the conformational
  state associated with that sequence sub-family.

This module only manipulates the MSA (.a3m); it does not run any structure prediction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class MsaRecord:
    header: str
    seq: str


def parse_a3m(path: str | Path) -> tuple[MsaRecord, list[MsaRecord]]:
    """Parse an a3m file. Returns (query_record, other_records). The query is the first entry."""
    records: list[MsaRecord] = []
    header = None
    seq_chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(MsaRecord(header, "".join(seq_chunks)))
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append(MsaRecord(header, "".join(seq_chunks)))
    if not records:
        raise ValueError(f"No sequences found in {path}")
    return records[0], records[1:]


def strip_inserts(seq: str) -> str:
    """Drop a3m lowercase insertion columns, keeping only match-state columns (upper/'-')."""
    return "".join(c for c in seq if not c.islower())


def write_a3m(path: str | Path, query: MsaRecord, members: list[MsaRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{query.header}\n{query.seq}\n")
        for rec in members:
            fh.write(f"{rec.header}\n{rec.seq}\n")


def hamming_distance_matrix(seqs: list[str]) -> np.ndarray:
    """Fractional mismatch distance over aligned, equal-length match-column sequences."""
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError("All sequences must be the same length (strip inserts first)")
    n = len(seqs)
    arr = np.array([list(s) for s in seqs])
    dist = np.zeros((n, n))
    for i in range(n):
        mismatches = (arr[i] != arr).mean(axis=1)
        dist[i] = mismatches
    return dist


def cluster_sequences(
    members: list[MsaRecord], distance_thresholds: list[float]
) -> dict[float, list[list[int]]]:
    """Hierarchical (average-linkage) clustering of homolog sequences at several distance
    cutoffs, in the spirit of AF-Cluster's multi-granularity scan. Returns, for each
    threshold, a list of clusters (each a list of indices into `members`).
    """
    if len(members) < 3:
        return {t: [list(range(len(members)))] for t in distance_thresholds}

    stripped = [strip_inserts(m.seq) for m in members]
    dist = hamming_distance_matrix(stripped)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")

    result: dict[float, list[list[int]]] = {}
    for t in distance_thresholds:
        labels = fcluster(z, t=t, criterion="distance")
        clusters: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(idx)
        result[t] = list(clusters.values())
    return result


def random_subsample(
    members: list[MsaRecord], fraction: float, min_seqs: int, rng: random.Random
) -> list[MsaRecord]:
    n = max(min_seqs, round(len(members) * fraction))
    n = min(n, len(members))
    return rng.sample(members, n) if n < len(members) else list(members)


def generate_diverse_msas(
    input_a3m: str | Path,
    out_dir: str | Path,
    random_fractions: list[float] = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
    n_random_replicates: int = 5,
    cluster_distance_thresholds: list[float] = (0.2, 0.35, 0.5),
    min_cluster_size: int = 3,
    min_seqs: int = 8,
    seed: int = 0,
) -> list[dict]:
    """Write a directory of subsampled/clustered a3m files derived from `input_a3m`.

    Returns a manifest: list of dicts with keys {path, method, depth, tag} describing
    each generated MSA, for downstream ColabFold batch runs.
    """
    query, members = parse_a3m(input_a3m)
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    manifest: list[dict] = []

    # Strategy 1: random shallow subsampling at several depths.
    for frac in random_fractions:
        for rep in range(n_random_replicates):
            subset = random_subsample(members, frac, min_seqs, rng)
            tag = f"rand_f{frac:.3f}_r{rep}"
            path = out_dir / f"{tag}.a3m"
            write_a3m(path, query, subset)
            manifest.append(
                {"path": str(path), "method": "random", "depth": len(subset), "tag": tag}
            )

    # Strategy 2: AF-Cluster-style sequence-family clustering, each cluster folded alone.
    clusters_by_threshold = cluster_sequences(members, list(cluster_distance_thresholds))
    for t, clusters in clusters_by_threshold.items():
        for ci, idxs in enumerate(clusters):
            if len(idxs) < min_cluster_size:
                continue
            subset = [members[i] for i in idxs]
            tag = f"clust_t{t:.2f}_c{ci}"
            path = out_dir / f"{tag}.a3m"
            write_a3m(path, query, subset)
            manifest.append(
                {"path": str(path), "method": "cluster", "depth": len(subset), "tag": tag}
            )

    return manifest
