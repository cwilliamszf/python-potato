"""Matplotlib plotting helpers, mirroring the figures produced by
FesCalc_Block.m / Plot_Imp_Variables.m / DSCcalc_Block.m."""

from __future__ import annotations

import numpy as np

from .alanine_scan import AlanineScanResult
from .coupling import CouplingResult
from .dsc import DSCResult
from .ionizable_network import IonizableNetworkResult
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


def plot_2d_landscape_surface(result: WSMEResult, ax=None, cmap="jet", vmin=None, vmax=None,
                               elev=28, azim=-55, colorbar=True, **kwargs):
    """3D free-energy surface (n_C vs n_N vs FE), in the style of the
    published GPCR-Landscapes figures: an angled jet-colormap surface with
    the fully-unfolded corner (0 structured blocks on both termini, which
    has zero population and so an undefined/infinite free energy) masked
    out rather than distorting the color scale."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")
    fes2D = np.where(np.isfinite(result.fes2D), result.fes2D, np.nan)
    nN, nC = np.meshgrid(np.arange(fes2D.shape[0]), np.arange(fes2D.shape[1]), indexing="ij")
    surf = ax.plot_surface(
        nC, nN, fes2D, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidth=0, antialiased=True, rcount=fes2D.shape[0], ccount=fes2D.shape[1], **kwargs,
    )
    ax.set_xlabel("n_C")
    ax.set_ylabel("n_N")
    ax.set_zlabel("FE (kJ/mol)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.55))
    if colorbar:
        plt.colorbar(surf, ax=ax, shrink=0.6, label="Free Energy (kJ/mol)")
    return ax


def plot_2d_landscape_surface_comparison(results_by_key: dict, cmap="jet", elev=28, azim=-55, figsize_per_panel=6):
    """One 3D surface subplot per entry in ``results_by_key`` (e.g. one per
    pH), all sharing a single color scale so panels are directly
    comparable -- matches send-to-user comparisons of the same receptor
    across conditions."""
    import matplotlib.pyplot as plt

    finite_vals = np.concatenate([
        r.fes2D[np.isfinite(r.fes2D)] for r in results_by_key.values()
    ])
    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())

    n = len(results_by_key)
    fig = plt.figure(figsize=(figsize_per_panel * n, figsize_per_panel))
    surf = None
    for i, (key, result) in enumerate(results_by_key.items()):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_2d_landscape_surface(result, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, elev=elev, azim=azim, colorbar=False)
        ax.set_title(str(key))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap),
        ax=fig.axes, shrink=0.6, label="Free Energy (kJ/mol)",
    )
    return fig


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


def plot_coupling_matrix(coupling: CouplingResult, ax=None, cmap="RdBu_r", vmax=None, colorbar=True, **kwargs):
    """Residue(block)-residue(block) coupling free-energy matrix (the
    'CouplingMat' in the original tool). Diverging colormap centered at
    zero: positive (red) = j and k tend to fold together, negative
    (blue) = folding one tends to unfold the other, near zero = no
    thermodynamic coupling. Pass ``vmax`` (e.g. the max over several
    CouplingResults) to put multiple panels on the same color scale."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    mat = coupling.coupling_free_energy
    if vmax is None:
        finite = mat[np.isfinite(mat)]
        vmax = np.nanmax(np.abs(finite)) if len(finite) else 1.0
    im = ax.pcolormesh(mat, cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax, **kwargs)
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Block Index")
    if colorbar:
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("Coupling Free Energy (kJ/mol)")
    return ax


def plot_comparison_grid(
    results_by_key: dict,
    coupling_by_key: dict = None,
    cmap="jet",
    coupling_cmap="RdBu_r",
    elev=28,
    azim=-55,
    figsize_per_panel=6.5,
):
    """Grid comparison across several runs (e.g. one column per pH):
    rows are 3D free-energy landscape, residue folding probability, and
    (if ``coupling_by_key`` is given) coupling free energy; columns are
    the entries of ``results_by_key`` (label -> WSMEResult). Each row
    shares one color scale across columns so they're directly comparable.
    """
    import matplotlib.pyplot as plt

    keys = list(results_by_key.keys())
    n = len(keys)
    n_rows = 3 if coupling_by_key else 2

    landscape_vals = np.concatenate([
        results_by_key[k].fes2D[np.isfinite(results_by_key[k].fes2D)] for k in keys
    ])
    lvmin, lvmax = float(landscape_vals.min()), float(landscape_vals.max())

    cvmax = None
    if coupling_by_key:
        coupling_vals = np.concatenate([
            coupling_by_key[k].coupling_free_energy[np.isfinite(coupling_by_key[k].coupling_free_energy)]
            for k in keys
        ])
        cvmax = float(np.max(np.abs(coupling_vals))) if len(coupling_vals) else 1.0

    fig = plt.figure(figsize=(figsize_per_panel * n, figsize_per_panel * n_rows))

    for col, key in enumerate(keys):
        result = results_by_key[key]

        ax1 = fig.add_subplot(n_rows, n, col + 1, projection="3d")
        plot_2d_landscape_surface(result, ax=ax1, cmap=cmap, vmin=lvmin, vmax=lvmax, elev=elev, azim=azim, colorbar=False)
        ax1.set_title(f"{key}\n3D Free Energy Landscape")

        ax2 = fig.add_subplot(n_rows, n, n + col + 1)
        plot_residue_folding_probability(result, ax=ax2, cmap=cmap)
        ax2.set_title("Residue Folding Probability")

        if coupling_by_key:
            ax3 = fig.add_subplot(n_rows, n, 2 * n + col + 1)
            plot_coupling_matrix(coupling_by_key[key], ax=ax3, cmap=coupling_cmap, vmax=cvmax)
            ax3.set_title("Coupling Free Energy")

    fig.tight_layout()
    return fig


_BURIAL_COLOR = {"buried": "#8B0000", "margin": "#DAA520", "exposed": "#4682B4"}


def plot_ionizable_network(result: IonizableNetworkResult, ax=None, label_his: bool = True, **kwargs):
    """3D scatter of ionizable residues (pHinder-style), edges from the
    trimmed Delaunay network, colored by approximate burial class.
    Histidines are labeled by author residue number by default, since
    they're the most likely pH-sensor candidates."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

    pos = result.position
    for i, j in result.edges:
        ax.plot(*zip(pos[i], pos[j]), color="gray", linewidth=0.6, alpha=0.6, zorder=1)

    colors = [_BURIAL_COLOR[c] for c in result.burial_class]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=40, zorder=2, **kwargs)

    if label_his:
        for i, rname in enumerate(result.resname):
            if rname == "HIS":
                ax.text(*pos[i], f"H{result.author_resnum[i]}", fontsize=8, zorder=3)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=k)
               for k, c in _BURIAL_COLOR.items()]
    ax.legend(handles=handles, title="Burial (approx.)")
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    return ax


def plot_mutational_response(scan_result: AlanineScanResult, ax=None, highlight: dict = None, **kwargs):
    """Mean +- std of the alanine-scanning mutational response (MR) per
    block -- Fig. 7b+c combined: how much a typical mutation anywhere in
    the structure perturbs each block's coupling, on average across every
    scanned site. ``highlight`` optionally maps {resnum: label} to mark
    specific mutated positions' own blocks with a vertical line."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    nb = len(scan_result.MR_mean)
    x = np.arange(nb)
    ax.fill_between(x, scan_result.MR_mean - scan_result.MR_std, scan_result.MR_mean + scan_result.MR_std,
                     color="lightgray", alpha=0.6, label="mean ± std")
    ax.plot(x, scan_result.MR_mean, color="k", linewidth=1.5, label="mean", **kwargs)
    ax.axhline(0, color="k", linewidth=0.5)
    if highlight:
        for resnum, label in highlight.items():
            b = scan_result.block_of_position.get(int(resnum))
            if b is not None:
                ax.axvline(b, color="crimson", linestyle=":", linewidth=1)
                ax.annotate(label, (b, ax.get_ylim()[1]), fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Block Index")
    ax.set_ylabel(r"Mutational Response, $\langle\Delta\Delta G^+\rangle$ (kJ/mol)")
    ax.legend()
    return ax


def plot_ddg_vs_distance(scan_result: AlanineScanResult, resnum: int, ax=None, **kwargs):
    """DeltaDeltaG+ vs. CA-CA distance from one mutated site -- Fig. 7d/f/h:
    shows whether a mutation's effect on coupling decays with distance or
    reaches far across the structure."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    dist, ddg = scan_result.ddg_vs_distance(resnum)
    finite = np.isfinite(dist) & np.isfinite(ddg)
    ax.scatter(dist[finite], ddg[finite], s=18, **kwargs)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel(r"C$\alpha$ Distance from Mutated Site (Å)")
    ax.set_ylabel(r"$\Delta\Delta G^+$ (kJ/mol)")
    ax.set_title(f"Perturbation response of residue {resnum}A")
    return ax


def plot_ddg_structure_map(scan_result: AlanineScanResult, resnum: int, ax=None, cmap="RdBu_r", **kwargs):
    """3D scatter of block CA centroids colored by DeltaDeltaG+ from one
    mutation -- a lightweight, dependency-free stand-in for the paper's
    PyMOL surface-colored structure renders (Fig. 7e/g/i)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
    pos = scan_result.block_ca_centroid
    v = scan_result.mean_ddg_vector[resnum]
    finite = np.isfinite(v)
    vmax = np.nanmax(np.abs(v)) if np.any(finite) else 1.0
    sc = ax.scatter(pos[finite, 0], pos[finite, 1], pos[finite, 2], c=v[finite], cmap=cmap,
                     vmin=-vmax, vmax=vmax, s=40, **kwargs)
    b = scan_result.block_of_position[resnum]
    ax.scatter(*pos[b], color="black", s=120, marker="*", label=f"mutated site (block {b})")
    plt.colorbar(sc, ax=ax, shrink=0.6, label=r"$\Delta\Delta G^+$ (kJ/mol)")
    ax.legend()
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
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


def plot_summary(result: WSMEResult, dsc_result: DSCResult = None, coupling_result: CouplingResult = None, save_path: str = None):
    """A single figure with the 1D profile, 2D landscape, residue folding
    probability, and (if provided) the DSC thermogram and coupling matrix
    -- similar in spirit to Plot_Imp_Variables.m."""
    import matplotlib.pyplot as plt

    n_panels = 3 + int(dsc_result is not None) + int(coupling_result is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    plot_1d_profile(result, ax=axes[0])
    axes[0].set_title("1D Free Energy Profile")

    plot_2d_landscape(result, ax=axes[1])
    axes[1].set_title("2D Free Energy Landscape")

    plot_residue_folding_probability(result, ax=axes[2])
    axes[2].set_title("Residue Folding Probability")

    next_ax = 3
    if dsc_result is not None:
        plot_dsc(dsc_result, ax=axes[next_ax])
        axes[next_ax].set_title("DSC Thermogram")
        next_ax += 1

    if coupling_result is not None:
        plot_coupling_matrix(coupling_result, ax=axes[next_ax])
        axes[next_ax].set_title("Coupling Free Energy")
        next_ax += 1

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
