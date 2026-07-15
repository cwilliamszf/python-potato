"""
Dimensionality reduction of a conformational ensemble's feature vectors,
reproducing Figure 3 of Fleetwood et al. 2021: PCA, MDS and t-SNE embeddings
of an ensemble, colored (downstream, in plotting.py) by free energy.

The paper's feature representation is "the inverse closest heavy atom
distances between residues" for a chosen residue-pair set -- i.e. the same
kind of distances computed in collective_variables.py, just used in bulk as
an unsupervised feature vector rather than as individually-interpreted CVs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE


def build_feature_matrix(features: Dict[str, np.ndarray]) -> Tuple[list, np.ndarray]:
    """Assemble a ``{feature_name: values}`` dict (e.g. one entry per
    residue-pair distance, each an array over conformers) into a
    ``(n_conformers, n_features)`` matrix. Returns ``(feature_names, matrix)``."""
    names = list(features.keys())
    if not names:
        raise ValueError("`features` is empty")
    matrix = np.column_stack([np.asarray(features[name], dtype=float) for name in names])
    return names, matrix


def inverse_distance_features(matrix: np.ndarray) -> np.ndarray:
    """Elementwise 1/d, matching the paper's feature representation."""
    matrix = np.asarray(matrix, dtype=float)
    if np.any(matrix == 0):
        raise ValueError("Cannot invert zero distances")
    return 1.0 / matrix


def pca_embedding(matrix: np.ndarray, n_components: int = 2, **kwargs):
    model = PCA(n_components=n_components, **kwargs)
    return model.fit_transform(matrix), model


def mds_embedding(matrix: np.ndarray, n_components: int = 2, random_state: int = 0, **kwargs):
    kwargs.setdefault("normalized_stress", "auto")
    kwargs.setdefault("init", "random")
    model = MDS(n_components=n_components, random_state=random_state, **kwargs)
    return model.fit_transform(matrix), model


def tsne_embedding(matrix: np.ndarray, n_components: int = 2, random_state: int = 0, perplexity: float = 30, **kwargs):
    n = matrix.shape[0]
    # t-SNE requires perplexity < n_samples; scale it down for small ensembles.
    perplexity = min(perplexity, max(1.0, (n - 1) / 3))
    model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, **kwargs)
    return model.fit_transform(matrix), model


_EMBEDDERS = {"pca": pca_embedding, "mds": mds_embedding, "tsne": tsne_embedding}


def embed(matrix: np.ndarray, method: str = "pca", **kwargs):
    """Dispatch to :func:`pca_embedding` / :func:`mds_embedding` /
    :func:`tsne_embedding` by name. Returns ``(embedding, fitted_model)``."""
    if method not in _EMBEDDERS:
        raise ValueError(f"Unknown method '{method}', expected one of {sorted(_EMBEDDERS)}")
    return _EMBEDDERS[method](matrix, **kwargs)
