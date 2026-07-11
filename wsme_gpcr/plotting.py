"""Matplotlib plotting helpers, mirroring the figures produced by
FesCalc_Block.m / Plot_Imp_Variables.m / DSCcalc_Block.m."""

from __future__ import annotations

import numpy as np

from .dsc import DSCResult
from .wsme import WSMEResult


def plot_1d_profile(result: WSMEResult, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.n_values, result.fes, color="k", linewidth=2, **kwargs)
    ax.set_xlabel("# of Structured Blocks")
    ax.set_ylabel("Free Energy (kJ/mol)")
    ax.set_xlim(result.n_values[0], result.n_values[-1])
    return ax


def plot_1d_profile_comparison(results_by_key: dict, ax=None, **kwargs):
    """Overlay several 1D free-energy profiles (e.g. one per pH) on one
    axes. ``results_by_key`` maps a label (e.g. a pH value) to a
    WSMEResult; note that different pH runs can have different
    block counts (contacts differ), so each curve is plotted against its
    own x-axis (fraction of structured blocks) to stay comparable."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for key, result in results_by_key.items():
        frac = result.n_values / result.n_values[-1]
        ax.plot(frac, result.fes, linewidth=2, label=str(key), **kwargs)
    ax.set_xlabel("Fraction of Structured Blocks")
    ax.set_ylabel("Free Energy (kJ/mol)")
    ax.set_xlim(0, 1)
    ax.legend(title="pH")
    return ax


def plot_2d_landscape(result: WSMEResult, ax=None, cmap="jet", **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    fes2D = np.where(np.isfinite(result.fes2D), result.fes2D, np.nan)
    im = ax.pcolormesh(fes2D, cmap=cmap, shading="auto", **kwargs)
    ax.set_xlabel("n_C-term (structured blocks)")
    ax.set_ylabel("n_N-term (structured blocks)")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Free Energy (kJ/mol)")
    return ax


def plot_2d_landscape_surface(result: WSMEResult, ax=None, cmap="jet", **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
    fes2D = np.where(np.isfinite(result.fes2D), result.fes2D, np.nan)
    nN, nC = np.meshgrid(np.arange(fes2D.shape[0]), np.arange(fes2D.shape[1]), indexing="ij")
    ax.plot_surface(nC, nN, fes2D, cmap=cmap, **kwargs)
    ax.set_xlabel("n_C-term")
    ax.set_ylabel("n_N-term")
    ax.set_zlabel("Free Energy (kJ/mol)")
    return ax


def plot_residue_folding_probability(result: WSMEResult, ax=None, cmap="jet", **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    # fpath is (n_values, block); transpose so blocks run along y like the
    # original 'Residue Index' axis.
    im = ax.pcolormesh(result.n_values, np.arange(result.fpath.shape[1]), result.fpath.T, cmap=cmap, shading="auto", **kwargs)
    ax.set_xlabel("# of Structured Blocks")
    ax.set_ylabel("Block Index")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Folding Probability")
    return ax


def plot_dsc(dsc_result: DSCResult, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dsc_result.T, dsc_result.Cp, color="k", linewidth=2, **kwargs)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$C_p$ (kJ mol$^{-1}$ K$^{-1}$)")
    ax.set_xlim(dsc_result.T[0], dsc_result.T[-1])
    return ax


def plot_summary(result: WSMEResult, dsc_result: DSCResult = None, save_path: str = None):
    """A single figure with the 1D profile, 2D landscape, residue folding
    probability, and (if provided) the DSC thermogram -- similar in spirit
    to Plot_Imp_Variables.m."""
    import matplotlib.pyplot as plt

    n_panels = 4 if dsc_result is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    plot_1d_profile(result, ax=axes[0])
    axes[0].set_title("1D Free Energy Profile")

    plot_2d_landscape(result, ax=axes[1])
    axes[1].set_title("2D Free Energy Landscape")

    plot_residue_folding_probability(result, ax=axes[2])
    axes[2].set_title("Residue Folding Probability")

    if dsc_result is not None:
        plot_dsc(dsc_result, ax=axes[3])
        axes[3].set_title("DSC Thermogram")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
