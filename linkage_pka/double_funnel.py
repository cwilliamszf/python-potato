"""Two-basin, pH-dependent free-energy landscape for GPR68 activation --
stitches WSME's per-conformer fold-order landscape (``wsme_gpcr``) with the
real inter-conformer free-energy offset (``linkage_pka``'s Wyman-linkage
machinery) into one continuous coordinate spanning both conformers.

Why this needs two separate tools, not just WSME alone
--------------------------------------------------------
``wsme_gpcr.wsme.run_wsme`` computes an exact (no MD, no sampling)
free-energy profile G(n) at one pH, where n = number of "folded" blocks
relative to ONE reference structure's own contact map. Critically, that
profile is normalized by *that pH's own total partition function*
(``fes_full = -R*T*log(fes_num/zfin)``, ``wsme_gpcr/wsme.py``) -- which
means the SHAPE of G(n) within one pH slice is a real, uncorrupted WSME
quantity (relative depths across n at fixed pH), but the ABSOLUTE
placement of one pH's curve relative to another pH's curve is not: each
pH's own zero-point has been independently reset. This is exactly the
same normalization ``linkage_pka/__init__.py``'s module docstring already
flags as the reason this pipeline exists as a tool separate from WSME.

This module restores that missing absolute placement using
``linkage_pka``'s own real inter-conformer free energy,
``delta_g_activation(pH)`` (e.g. from ``linkage.delta_g_act_from_ln_z`` on
a coupled cluster's ``ln_z_total``) -- anchoring each conformer's WSME
curve at its own REFERENCE structure (n=nblocks, the actual
crystallographic/predicted conformation the PB calculation was run on,
not necessarily wherever WSME's own free energy happens to be lowest),
then letting the rest of each conformer's WSME-computed local
folded/unfolded shape follow from there unmodified.

Calibration status
-------------------
As of this writing, ``delta_g_activation`` inputs sourced from
``linkage_pka``'s Poisson-Boltzmann pKa's have FAILED Gate A (real
SNase-benchmark RMSE 7.9-11.3 pKa units against a 1.0-unit acceptance
threshold in every relaxation variant tried -- see ``gate_a.py`` and
``FINDINGS.md``). Any landscape built from that anchor is a
pipeline-mechanics demonstration of what this stitching would show once
calibrated, not a validated scientific prediction of GPR68's real
activation thermodynamics -- report it with that caveat every time, per
this project's own documentation discipline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DoubleFunnelResult:
    ph_values: np.ndarray     # (n_ph,)
    q_values: np.ndarray      # (n_q,) signed coordinate: [-1, 0) inactive basin, (0, +1] active basin
    free_energy: np.ndarray   # (n_ph, n_q) kJ/mol, real cross-pH AND cross-conformer absolute scale
    q_gap_index: int          # free_energy[:, :q_gap_index] is the inactive basin, [:, q_gap_index:] the active basin


def build_double_funnel_landscape(fes_inactive, n_values_inactive, fes_active, n_values_active,
                                   delta_g_activation, ph_values) -> DoubleFunnelResult:
    """Stitch two WSME per-conformer, per-pH free-energy profiles into one
    two-basin landscape, anchored by the real inter-conformer offset.

    ``fes_inactive``, ``fes_active``: (n_ph, nblocks) kJ/mol -- one
    ``WSMEResult.fes`` row per pH, for the inactive/active reference
    structures respectively (e.g. stack ``wsme_gpcr.pipeline.
    run_pipeline_multi_ph``'s per-pH ``result.fes`` in ``ph_values`` order).
    ``n_values_inactive``, ``n_values_active``: (nblocks,) -- the matching
    ``WSMEResult.n_values`` (assumed the same block count/order at every
    pH for that conformer; ``run_pipeline_multi_ph`` re-derives blocks per
    pH, so this is only exactly true if the contact map/blocking is
    pH-stable for the pH range used -- not independently checked here).
    ``delta_g_activation``: (n_ph,) kJ/mol, G_active - G_inactive between
    the two conformers' REFERENCE (fully-resolved, n=nblocks) structures at
    each pH -- e.g. ``linkage.delta_g_act_from_ln_z``'s output. This is the
    one piece of information WSME's own normalization cannot supply (see
    module docstring); it must come from an independent absolute free
    energy calculation on the same two reference structures.
    ``ph_values``: (n_ph,) -- must be the same grid as the three arrays
    above; this function does not resample or interpolate mismatched grids.

    Anchoring: each conformer's WSME curve is shifted so its OWN reference
    state (n=nblocks, the last entry of its ``n_values`` -- i.e. the actual
    structure the PB calculation was run on, not wherever WSME's own free
    energy happens to be lowest) sits at 0 (inactive) or
    ``delta_g_activation(pH)`` (active). The rest of each conformer's
    WSME-computed local unfolding shape is preserved unmodified relative
    to that anchor.

    Coordinate: block counts differ between conformers (independently
    derived contact maps), so each is first converted to a fractional
    coordinate ``n/nblocks`` (matching ``wsme_gpcr.plotting.
    plot_1d_profile_comparison``'s existing cross-run convention). Each
    conformer's own reference state (frac=1) is placed adjacent to
    ``Q=0`` -- the two real, PB-anchored structures being compared sit at
    the coordinate's center -- with each conformer's fully-disordered end
    (frac->0, n=1) at the outer edges (Q=-1, Q=+1). ``Q=0`` itself is NOT
    a computed transition state: there is no WSME or PB calculation
    connecting the two basins directly, so nothing at or near the seam
    between the two halves of ``free_energy`` should be read as a
    computed barrier -- see ``q_gap_index``.
    """
    ph_values = np.asarray(ph_values, dtype=float)
    delta_g_activation = np.asarray(delta_g_activation, dtype=float)
    fes_inactive = np.asarray(fes_inactive, dtype=float)
    fes_active = np.asarray(fes_active, dtype=float)
    n_values_inactive = np.asarray(n_values_inactive, dtype=float)
    n_values_active = np.asarray(n_values_active, dtype=float)

    n_ph = len(ph_values)
    if not (fes_inactive.shape[0] == fes_active.shape[0] == delta_g_activation.shape[0] == n_ph):
        raise ValueError(
            "fes_inactive, fes_active, delta_g_activation, and ph_values must share the same "
            f"pH-axis length; got {fes_inactive.shape[0]}, {fes_active.shape[0]}, "
            f"{delta_g_activation.shape[0]}, {n_ph}"
        )
    if fes_inactive.shape[1] != len(n_values_inactive):
        raise ValueError("fes_inactive's block axis must match len(n_values_inactive)")
    if fes_active.shape[1] != len(n_values_active):
        raise ValueError("fes_active's block axis must match len(n_values_active)")
    if len(n_values_inactive) < 2 or len(n_values_active) < 2:
        raise ValueError("each conformer needs at least 2 block-count values to form a coordinate")

    # Fraction of structured blocks, ascending (n=1 -> ~0, n=nblocks -> 1.0).
    frac_inactive = n_values_inactive / n_values_inactive[-1]
    frac_active = n_values_active / n_values_active[-1]

    # Inactive: disordered (frac~0) at Q=-1, reference (frac=1) at Q=0- -- already ascending, no reorder needed.
    q_inactive = frac_inactive - 1.0
    shifted_inactive = fes_inactive - fes_inactive[:, -1:]

    # Active: reference (frac=1) at Q=0+, disordered (frac~0) at Q=+1 --
    # requires reversing frac_active's natural ascending order (and the
    # matching fes columns) to keep the final Q axis monotonically increasing.
    q_active = 1.0 - frac_active[::-1]
    shifted_active = (fes_active - fes_active[:, -1:] + delta_g_activation[:, None])[:, ::-1]

    q_values = np.concatenate([q_inactive, q_active])
    free_energy = np.concatenate([shifted_inactive, shifted_active], axis=1)
    q_gap_index = len(q_inactive)

    return DoubleFunnelResult(ph_values=ph_values, q_values=q_values,
                               free_energy=free_energy, q_gap_index=q_gap_index)


def plot_double_funnel(result: DoubleFunnelResult, ax=None, **kwargs):
    """2D heatmap of ``result.free_energy`` over (pH, Q), with a vertical
    line marking ``q_gap_index`` (the non-computed seam between basins --
    see ``build_double_funnel_landscape``'s docstring). Mirrors
    ``wsme_gpcr.plotting``'s style; use
    ``wsme_gpcr.plotting.save_figure(ax.figure, path)`` to write PNG+SVG.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(result.q_values, result.ph_values, result.free_energy,
                          shading="nearest", cmap="viridis", **kwargs)
    seam_q = result.q_values[result.q_gap_index - 1:result.q_gap_index + 1].mean()
    ax.axvline(seam_q, color="white", linestyle="--", linewidth=1.5,
               label="basin seam (not a computed barrier)")
    ax.set_xlabel("Q  (inactive basin <  0  < active basin)")
    ax.set_ylabel("pH")
    ax.legend(loc="upper right", fontsize=8)
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label("Free energy (kJ/mol)")
    return ax
