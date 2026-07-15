"""
Free-energy landscape estimation along one or two collective variables (CVs).

Two ways of turning an ensemble into a landscape are supported, matching the
two kinds of input this pipeline can realistically receive from tool 3:

* ``gibbs``: each conformer already carries its own (absolute or relative)
  Gibbs free energy estimate (e.g. MM-GBSA/FEP per representative AlphaFold
  conformer). Free energies of conformers that land in the same region of CV
  space are combined via the Boltzmann-weighted partition function, i.e.
  properly summing over microstates rather than averaging energies:

      G(bin) = -RT * ln( sum_i w_i * exp(-G_i / RT) )

* ``counts`` / ``weighted``: conformers are (optionally weighted) samples
  from a Boltzmann ensemble (e.g. an adaptive-sampling swarm), and the
  landscape is the usual population-density estimate:

      G(bin) = -RT * ln( P(bin) )

  as used directly in Fleetwood et al. 2021.

Both 1D and 2D landscapes are estimated with a (weighted) Gaussian KDE by
default, which is far less noisy than raw histograms for sparse ensembles
(tens-hundreds of conformers, as typically produced by AlphaFold-based
sampling) -- a plain weighted-histogram estimator is also available via
``method="histogram"``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import gaussian_kde

KB_KCAL_PER_MOL_K = 0.0019872041  # Boltzmann constant, kcal/(mol*K)


def _rt(temperature: float) -> float:
    return KB_KCAL_PER_MOL_K * temperature


def boltzmann_weights(gibbs: np.ndarray, temperature: float = 310.0) -> np.ndarray:
    """Normalized Boltzmann weights w_i = exp(-(G_i - G_min)/RT) / Z for a set
    of per-conformer Gibbs free energies."""
    gibbs = np.asarray(gibbs, dtype=float)
    if gibbs.size == 0:
        return gibbs
    rt = _rt(temperature)
    g_min = np.nanmin(gibbs)
    w = np.exp(-(gibbs - g_min) / rt)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Boltzmann weights failed to normalize (check `gibbs`/`temperature`)")
    return w / total


def free_energy_from_gibbs(
    gibbs: np.ndarray, weights: Optional[np.ndarray] = None, temperature: float = 310.0
) -> float:
    """Combine a set of microstate Gibbs free energies into a single
    macrostate free energy via the Boltzmann-weighted partition function.

    ``weights`` are extra (e.g. sampling) weights multiplying each
    microstate's Boltzmann factor; pass ``None`` to weight all microstates
    equally.
    """
    gibbs = np.asarray(gibbs, dtype=float)
    if gibbs.size == 0:
        return float("nan")
    w = np.ones_like(gibbs) if weights is None else np.asarray(weights, dtype=float)
    rt = _rt(temperature)
    g_min = gibbs.min()
    with np.errstate(over="ignore"):
        partition = np.sum(w * np.exp(-(gibbs - g_min) / rt))
    if partition <= 0 or not np.isfinite(partition):
        return float("nan")
    return float(g_min - rt * np.log(partition))


def _resolve_weights(
    n: int,
    gibbs: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    temperature: float,
) -> np.ndarray:
    """Turn (gibbs, weights) into a single set of normalized sample weights
    used to drive the KDE / histogram density estimate."""
    if gibbs is not None:
        w = boltzmann_weights(np.asarray(gibbs, dtype=float), temperature)
        if weights is not None:
            w = w * np.asarray(weights, dtype=float)
            w = w / w.sum()
        return w
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        return w / w.sum()
    return np.full(n, 1.0 / n)


def landscape_1d(
    cv: np.ndarray,
    gibbs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    temperature: float = 310.0,
    method: str = "kde",
    grid_size: int = 200,
    bins: int = 40,
    bandwidth=None,
    pad: float = 0.1,
    min_count: int = 1,
) -> Dict[str, np.ndarray]:
    """Free-energy landscape along a single CV.

    Returns a dict with ``cv`` (grid points) and ``dG`` (kcal/mol, shifted so
    the global minimum is 0), plus ``density``/``counts`` depending on method.
    """
    cv = np.asarray(cv, dtype=float)
    n = len(cv)
    if n == 0:
        raise ValueError("`cv` is empty")
    rt = _rt(temperature)
    w = _resolve_weights(n, gibbs, weights, temperature)

    if method == "kde":
        span = cv.max() - cv.min()
        span = span if span > 0 else 1.0
        lo, hi = cv.min() - pad * span, cv.max() + pad * span
        grid = np.linspace(lo, hi, grid_size)
        kde = gaussian_kde(cv, weights=w, bw_method=bandwidth)
        density = np.clip(kde(grid), 1e-300, None)
        dG = -rt * np.log(density)
        dG -= dG.min()
        return {"cv": grid, "dG": dG, "density": density, "method": method}

    if method == "histogram":
        edges = np.histogram_bin_edges(cv, bins=bins)
        idx = np.clip(np.digitize(cv, edges[1:-1]), 0, len(edges) - 2)
        centers = 0.5 * (edges[:-1] + edges[1:])
        nbins = len(centers)
        dG = np.full(nbins, np.nan)
        counts = np.zeros(nbins, dtype=int)
        for b in range(nbins):
            mask = idx == b
            counts[b] = int(mask.sum())
            if counts[b] < min_count:
                continue
            if gibbs is not None:
                sub_weights = weights[mask] if weights is not None else None
                dG[b] = free_energy_from_gibbs(np.asarray(gibbs)[mask], sub_weights, temperature)
            else:
                p = w[mask].sum()
                dG[b] = -rt * np.log(p) if p > 0 else np.nan
        finite = np.isfinite(dG)
        if finite.any():
            dG = dG - np.nanmin(dG[finite])
        return {"cv": centers, "edges": edges, "dG": dG, "counts": counts, "method": method}

    raise ValueError(f"Unknown method '{method}', expected 'kde' or 'histogram'")


def landscape_2d(
    cv_x: np.ndarray,
    cv_y: np.ndarray,
    gibbs: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    temperature: float = 310.0,
    method: str = "kde",
    grid_size: int = 100,
    bins: int = 30,
    bandwidth=None,
    pad: float = 0.1,
    min_count: int = 1,
) -> Dict[str, np.ndarray]:
    """Free-energy landscape along two CVs, analogous to Figure 2b of
    Fleetwood et al. 2021. Returns a dict with meshgrid ``X``, ``Y`` and
    ``dG`` (kcal/mol, shifted so the global minimum is 0)."""
    cv_x = np.asarray(cv_x, dtype=float)
    cv_y = np.asarray(cv_y, dtype=float)
    n = len(cv_x)
    if n == 0 or len(cv_y) != n:
        raise ValueError("`cv_x` and `cv_y` must be the same non-zero length")
    rt = _rt(temperature)
    w = _resolve_weights(n, gibbs, weights, temperature)

    def _grid_1d(values):
        span = values.max() - values.min()
        span = span if span > 0 else 1.0
        lo, hi = values.min() - pad * span, values.max() + pad * span
        return np.linspace(lo, hi, grid_size)

    if method == "kde":
        xs, ys = _grid_1d(cv_x), _grid_1d(cv_y)
        X, Y = np.meshgrid(xs, ys)
        kde = gaussian_kde(np.vstack([cv_x, cv_y]), weights=w, bw_method=bandwidth)
        density = np.clip(kde(np.vstack([X.ravel(), Y.ravel()])), 1e-300, None).reshape(X.shape)
        dG = -rt * np.log(density)
        dG -= dG.min()
        return {"x": xs, "y": ys, "X": X, "Y": Y, "dG": dG, "density": density, "method": method}

    if method == "histogram":
        x_edges = np.histogram_bin_edges(cv_x, bins=bins)
        y_edges = np.histogram_bin_edges(cv_y, bins=bins)
        x_idx = np.clip(np.digitize(cv_x, x_edges[1:-1]), 0, len(x_edges) - 2)
        y_idx = np.clip(np.digitize(cv_y, y_edges[1:-1]), 0, len(y_edges) - 2)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers)
        dG = np.full(X.shape, np.nan)
        counts = np.zeros(X.shape, dtype=int)
        for iy in range(len(y_centers)):
            for ix in range(len(x_centers)):
                mask = (x_idx == ix) & (y_idx == iy)
                counts[iy, ix] = int(mask.sum())
                if counts[iy, ix] < min_count:
                    continue
                if gibbs is not None:
                    sub_weights = weights[mask] if weights is not None else None
                    dG[iy, ix] = free_energy_from_gibbs(np.asarray(gibbs)[mask], sub_weights, temperature)
                else:
                    p = w[mask].sum()
                    dG[iy, ix] = -rt * np.log(p) if p > 0 else np.nan
        finite = np.isfinite(dG)
        if finite.any():
            dG = dG - np.nanmin(dG[finite])
        return {"x": x_centers, "y": y_centers, "X": X, "Y": Y, "dG": dG, "counts": counts, "method": method}

    raise ValueError(f"Unknown method '{method}', expected 'kde' or 'histogram'")
