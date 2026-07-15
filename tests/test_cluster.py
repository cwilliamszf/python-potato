import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpcr_ensemble import cluster as clust


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_kabsch_rmsd_zero_for_rotated_translated_copy():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 3))
    rot = random_rotation(rng)
    transformed = coords @ rot.T + np.array([5.0, -3.0, 2.0])
    rmsd = clust.kabsch_rmsd(coords, transformed)
    assert rmsd < 1e-6


def test_kabsch_rmsd_positive_for_different_structures():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20, 3))
    b = rng.normal(size=(20, 3))
    assert clust.kabsch_rmsd(a, b) > 1.0


def test_cluster_by_rmsd_separates_two_basins():
    # kabsch_rmsd is translation/rotation invariant (as it must be for comparing
    # conformations), so a genuinely different "basin" needs an internal shape change,
    # not a rigid-body shift of the whole point cloud -- mimic a TM6-swing-like
    # displacement of a subset of residues relative to the rest.
    rng = np.random.default_rng(2)
    basin_a = rng.normal(size=(15, 3))
    basin_b = basin_a.copy()
    basin_b[-5:] += np.array([15.0, 0.0, 0.0])

    coord_sets = []
    for _ in range(6):
        coord_sets.append(basin_a + rng.normal(scale=0.1, size=basin_a.shape))
    for _ in range(6):
        coord_sets.append(basin_b + rng.normal(scale=0.1, size=basin_b.shape))

    rmsd_matrix = clust.pairwise_rmsd_matrix(coord_sets)
    labels = clust.cluster_by_rmsd(rmsd_matrix, distance_cutoff=2.0)
    assert len(set(labels)) == 2
    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_select_representatives_picks_highest_quality():
    labels = np.array([1, 1, 2, 2])
    quality = [50.0, 90.0, 70.0, 60.0]
    reps = clust.select_representatives(labels, quality)
    assert reps[1] == 1  # index 1 has higher quality within cluster 1
    assert reps[2] == 2  # index 2 has higher quality within cluster 2


def test_classical_mds_preserves_pairwise_order():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(10, 3))
    true_dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    embedding = clust.classical_mds(true_dist, n_components=2)
    embed_dist = np.linalg.norm(embedding[:, None, :] - embedding[None, :, :], axis=-1)
    # correlation between true and embedded distances should be strong
    corr = np.corrcoef(true_dist.ravel(), embed_dist.ravel())[0, 1]
    assert corr > 0.8


def test_common_core_coords_intersects_residue_sets():
    d1 = {1: np.array([0, 0, 0]), 2: np.array([1, 0, 0]), 3: np.array([2, 0, 0]), 4: np.array([3, 0, 0])}
    d2 = {2: np.array([1, 1, 1]), 3: np.array([2, 1, 1]), 4: np.array([3, 1, 1]), 5: np.array([4, 1, 1])}
    out = clust.common_core_coords([d1, d2])
    assert out[0].shape == (3, 3)  # residues 2,3,4 shared
