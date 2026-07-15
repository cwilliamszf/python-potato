"""
Matplotlib plotting for free-energy landscapes and embeddings.

Color always encodes the same quantity throughout this module -- Gibbs free
energy magnitude (kcal/mol relative to the ensemble's minimum) -- so it uses
a single perceptually-uniform sequential colormap (``viridis``) everywhere
rather than a rainbow/jet scale, consistent for both the continuous
landscape surfaces and the discrete per-conformer scatter overlays.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

DG_LABEL = r"$\Delta G$ (kcal/mol)"
SEQUENTIAL_CMAP = "viridis"


def plot_1d_landscape(
    landscape: Dict,
    cv_label: str = "Collective variable",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
    """1D free-energy profile, e.g. Figure 2a style."""
    ax = ax or plt.gca()
    ax.plot(landscape["cv"], landscape["dG"], lw=2, label=label, **kwargs)
    ax.set_xlabel(cv_label)
    ax.set_ylabel(DG_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if label:
        ax.legend(frameon=False)
    return ax


def plot_2d_landscape(
    landscape: Dict,
    x_label: str = "CV 1",
    y_label: str = "CV 2",
    ax: Optional[plt.Axes] = None,
    levels: int = 20,
    scatter: Optional[pd.DataFrame] = None,
    scatter_x: Optional[str] = None,
    scatter_y: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
    """2D free-energy landscape as a filled contour, e.g. Figure 2b style.
    Optionally overlays the raw conformer positions as a scatter (pass
    ``scatter``/``scatter_x``/``scatter_y`` to show individual structures on
    top of the smoothed landscape)."""
    ax = ax or plt.gca()
    cs = ax.contourf(landscape["X"], landscape["Y"], landscape["dG"], levels=levels, cmap=SEQUENTIAL_CMAP, **kwargs)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label(DG_LABEL)
    if scatter is not None:
        ax.scatter(
            scatter[scatter_x], scatter[scatter_y], s=14, facecolor="white", edgecolor="black", linewidth=0.5, alpha=0.85
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax


def plot_embedding(
    embedding_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = "gibbs_kcal_mol",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Scatter of the ensemble in embedding space, colored by free energy --
    Figure 3 style, but colored continuously by dG rather than by ligand."""
    ax = ax or plt.gca()
    color_values = embedding_df[color_col] - embedding_df[color_col].min()
    sc = ax.scatter(
        embedding_df[x_col], embedding_df[y_col], c=color_values, cmap=SEQUENTIAL_CMAP, s=28, edgecolor="none", **kwargs
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(DG_LABEL)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
