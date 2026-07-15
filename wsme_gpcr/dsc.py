"""Differential scanning calorimetry (DSC) thermogram calculation.

Ports ``DSCcalc_Block.m``: sweep temperature, track how the total
partition function Z(T) changes, and turn its log-derivatives into an
excess heat-capacity curve, plus an empirical native-state heat-capacity
baseline (Mw-scaled) added on top -- matching what's plotted against
experimental DSC thermograms in the original tool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .blocking import BlockModel
from .structure import Structure
from .wsme import WSMEParams, _build_topology, partition_function


@dataclass
class DSCResult:
    T: np.ndarray  # (n,) K
    Cp: np.ndarray  # (n,) kJ/mol/K, total (excess + native baseline)
    Cp_excess: np.ndarray  # (n,) kJ/mol/K, from the partition function alone
    logZ: np.ndarray  # (n,) log of the total partition function at each T


def compute_dsc(
    structure: Structure,
    block_model: BlockModel,
    ss_mask: np.ndarray,
    params: WSMEParams = None,
    T_grid: np.ndarray = None,
    baseline_intercept: float = 1.6,
    baseline_slope: float = 6.7e-3,
) -> DSCResult:
    """Compute a DSC thermogram by sweeping ``params.T`` over ``T_grid``.

    ``baseline_intercept``/``baseline_slope`` set the empirical native-state
    heat-capacity baseline (J/g/K at 273.15 K, and its slope with
    temperature) that's added on top of the excess heat capacity derived
    from the partition function; 1.6 and 6.7e-3 are the values used for
    GPCRs in DSCcalc_Block.m.
    """
    if params is None:
        params = WSMEParams()
    if T_grid is None:
        T_grid = np.arange(273.0, 374.0, 1.0)
    T_grid = np.asarray(T_grid, dtype=float)

    topo = _build_topology(block_model)

    logZ = np.empty(len(T_grid))
    for i, T in enumerate(T_grid):
        p = WSMEParams(**{**params.__dict__, "T": float(T)})
        z = partition_function(structure, block_model, ss_mask, p, topo=topo)
        logZ[i] = np.log(z)

    R = params.R
    Tint = np.arange(T_grid[0] - 10.0, T_grid[-1] + 10.0 + 1e-9, 0.1)

    spline_logZ = CubicSpline(T_grid, logZ, extrapolate=True)
    logZint = spline_logZ(Tint)

    der1d = np.diff(logZint) / np.diff(Tint)
    mid_T = Tint[:-1]
    der1df = CubicSpline(mid_T, der1d, extrapolate=True)(T_grid)

    der1d2 = CubicSpline(T_grid, der1df, extrapolate=True)(Tint)
    der2d = np.diff(der1d2) / np.diff(Tint)
    der2df = CubicSpline(mid_T, der2d, extrapolate=True)(T_grid)

    Cp_excess = 2 * R * T_grid * der1df + R * T_grid ** 2 * der2df

    Mw = structure.nres * 110.0
    baseline = (baseline_intercept + baseline_slope * (T_grid - 273.15)) * Mw / 1000.0
    Cp = Cp_excess + baseline

    return DSCResult(T=T_grid, Cp=Cp, Cp_excess=Cp_excess, logZ=logZ)
