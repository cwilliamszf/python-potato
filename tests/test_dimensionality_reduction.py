import numpy as np
import pytest

from gpcr_energy_landscapes.dimensionality_reduction import (
    build_feature_matrix,
    embed,
    inverse_distance_features,
)


def test_build_feature_matrix_shape_and_order():
    names, matrix = build_feature_matrix({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert names == ["a", "b"]
    assert matrix.shape == (3, 2)
    np.testing.assert_array_equal(matrix[:, 0], [1, 2, 3])


def test_build_feature_matrix_empty_raises():
    with pytest.raises(ValueError):
        build_feature_matrix({})


def test_inverse_distance_features():
    matrix = np.array([[1.0, 2.0], [4.0, 0.5]])
    inv = inverse_distance_features(matrix)
    np.testing.assert_allclose(inv, np.array([[1.0, 0.5], [0.25, 2.0]]))


def test_inverse_distance_features_rejects_zero():
    with pytest.raises(ValueError):
        inverse_distance_features(np.array([[0.0, 1.0]]))


def test_pca_embedding_separates_two_clusters():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0, 0, 0], scale=0.1, size=(20, 3))
    cluster_b = rng.normal(loc=[10, 10, 10], scale=0.1, size=(20, 3))
    matrix = np.vstack([cluster_a, cluster_b])
    embedding, model = embed(matrix, method="pca", n_components=2)
    assert embedding.shape == (40, 2)
    # first PC should separate the two well-separated clusters
    pc1_a, pc1_b = embedding[:20, 0], embedding[20:, 0]
    assert abs(pc1_a.mean() - pc1_b.mean()) > 5 * max(pc1_a.std(), pc1_b.std())


def test_mds_and_tsne_embedding_shapes():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(15, 4))
    mds_emb, _ = embed(matrix, method="mds", n_components=2)
    tsne_emb, _ = embed(matrix, method="tsne", n_components=2, perplexity=5)
    assert mds_emb.shape == (15, 2)
    assert tsne_emb.shape == (15, 2)


def test_embed_unknown_method_raises():
    with pytest.raises(ValueError):
        embed(np.zeros((5, 2)), method="bogus")
