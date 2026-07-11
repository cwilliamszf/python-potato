"""Wyman/Tanford proton-linkage thermodynamics: given per-residue pKa's on
two conformational states (active/inactive), compute the pH-dependence of
the conformational equilibrium and the linked proton uptake -- no sampling,
no molecular dynamics, exact given the pKa's.

Reference: Wyman, J. (1964) "Linked functions and reciprocal effects in
hemoglobin: a second look." Adv. Protein Chem. 19:223-286. Tanford, C.
(1970) "Protein denaturation. Part C." Adv. Protein Chem. 24:1-95.

A note on the per-residue occupancy function
----------------------------------------------
For a two-state process P_inactive <-> P_active with equilibrium constant
K(pH) = [active]/[inactive], Wyman's linked-function theorem gives the
number of protons taken up on activation as

    Delta_n_H(pH) = d ln K(pH) / d ln[H+]

Writing ln K(pH) = -DeltaG_act(pH)/RT and DeltaG_act(pH) = DeltaG_int -
RT * sum_i [ln(1+10^(pKa_i,active-pH)) - ln(1+10^(pKa_i,inactive-pH))]
(the standard multi-site linkage factorization -- see ``delta_g_activation``
below), differentiating and converting d/d(pH) to d/d(ln[H+]) = -1/ln(10) *
d/d(pH) gives, for every site i regardless of whether it is chemically an
acid or a base,

    Delta_n_H(pH) = sum_i [theta_i,active(pH) - theta_i,inactive(pH)]
    theta_i(pH) = 1 / (1 + 10^(pH - pKa_i))      (fraction protonated)

theta_i(pH) is the *fraction protonated* (occupancy of the conjugate-acid
form) of site i, used identically for acidic residues (Asp/Glu/Tyr, whose
protonated form is neutral) and basic residues (His/Lys/Arg, whose
protonated form is cationic) -- the acid/base chemistry determines which
net charge state theta=1 corresponds to (relevant for the electrostatics
and structure-prep steps elsewhere in the pipeline), not the functional
form of theta(pH) itself. Using "fraction ionized" (1/(1+10^(pKa-pH))) for
acidic sites here, instead of fraction protonated, would flip the sign of
their contribution to Delta_n_H and silently break the Wyman-relation
identity checked in ``tests/test_linkage.py`` -- verified directly by
differentiating ``delta_g_activation`` symbolically and confirming it
reduces to this uniform theta(pH), and again numerically (finite-difference
d(DeltaG_act/RT)/d(ln[H+]) against the closed-form Delta_n_H) in the test
suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_KJ_PER_MOL_K = 8.31446261815324e-3  # kJ/(mol*K)
LN10 = np.log(10.0)


def protonation_fraction(ph, pka):
    """Fraction of a titratable site in its protonated (conjugate-acid)
    form at the given pH, given its pKa -- theta(pH) = 1/(1+10^(pH-pKa)).
    Vectorized; numerically stable for extreme (pH-pKa) via scipy's expit
    (logistic sigmoid): theta(pH) = expit(-(pH-pKa)*ln10)."""
    from scipy.special import expit

    ph = np.asarray(ph, dtype=float)
    pka = np.asarray(pka, dtype=float)
    return expit(-(ph - pka) * LN10)


def _log1p_10pow(x):
    """ln(1 + 10^x), numerically stable for large |x| via logaddexp
    (avoids overflow in 10**x for x >> 1 and precision loss for x << -1)."""
    x = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, x * LN10)


@dataclass
class LinkageResult:
    ph: np.ndarray                  # (n_ph,)
    delta_g_act: np.ndarray         # (n_ph,) kJ/mol -- pH-dependent part of the activation free energy
    delta_n_h: np.ndarray           # (n_ph,) net protons taken up on activation
    delta_n_h_per_residue: np.ndarray  # (n_ph, n_sites) per-site contribution to delta_n_h
    resnums: np.ndarray             # (n_sites,) author resnums, aligned to the per-residue axis
    pka_active: np.ndarray          # (n_sites,)
    pka_inactive: np.ndarray        # (n_sites,)

    def top_contributors(self, ph_value: float, n: int = 10) -> list:
        """Residues ranked by |contribution to Delta_n_H| at the nearest
        computed pH to ``ph_value`` -- "which positions carry the proton
        uptake" (pipeline spec, step 5)."""
        i = int(np.argmin(np.abs(self.ph - ph_value)))
        contrib = self.delta_n_h_per_residue[i]
        order = np.argsort(-np.abs(contrib))[:n]
        return [(int(self.resnums[j]), float(contrib[j])) for j in order]


def compute_linkage(
    ph_values,
    pka_active: dict,
    pka_inactive: dict,
    T: float = 298.15,
) -> LinkageResult:
    """Compute DeltaDeltaG_act(pH) and Delta_n_H(pH) from per-residue pKa's
    on the active and inactive conformers.

    ``pka_active``/``pka_inactive`` map author resnum -> pKa (float), or ->
    NaN/None for a site whose pKa could not be resolved on that structure
    (e.g. calculation failed to converge) -- such sites are excluded from
    the sums with a warning-worthy NaN propagated into their per-residue
    column, not silently dropped, so a caller can see exactly which sites
    were unresolved rather than have them vanish.

    Only resnums present in *both* dicts are used (a site whose pKa is
    only known in one conformer cannot contribute a difference); this is
    the caller's responsibility to arrange (e.g. by running the pKa
    calculation on the same ionizable-residue list for both structures).
    """
    resnums = sorted(set(pka_active) & set(pka_inactive))
    if not resnums:
        raise ValueError("no residues with pKa values in both pka_active and pka_inactive")

    pka_a = np.array([np.nan if pka_active[r] is None else float(pka_active[r]) for r in resnums])
    pka_i = np.array([np.nan if pka_inactive[r] is None else float(pka_inactive[r]) for r in resnums])

    ph = np.asarray(list(ph_values), dtype=float)
    RT = R_KJ_PER_MOL_K * T

    # DeltaDeltaG_act(pH) = -RT * [sum_i ln(1+10^(pKa_i,active-pH)) - sum_i ln(1+10^(pKa_i,inactive-pH))]
    log_term_active = _log1p_10pow(pka_a[None, :] - ph[:, None])      # (n_ph, n_sites)
    log_term_inactive = _log1p_10pow(pka_i[None, :] - ph[:, None])    # (n_ph, n_sites)
    delta_g_act = -RT * (np.nansum(log_term_active, axis=1) - np.nansum(log_term_inactive, axis=1))

    theta_active = protonation_fraction(ph[:, None], pka_a[None, :])      # (n_ph, n_sites)
    theta_inactive = protonation_fraction(ph[:, None], pka_i[None, :])    # (n_ph, n_sites)
    delta_n_h_per_residue = theta_active - theta_inactive
    delta_n_h = np.nansum(delta_n_h_per_residue, axis=1)

    return LinkageResult(
        ph=ph,
        delta_g_act=delta_g_act,
        delta_n_h=delta_n_h,
        delta_n_h_per_residue=delta_n_h_per_residue,
        resnums=np.array(resnums),
        pka_active=pka_a,
        pka_inactive=pka_i,
    )


def delta_n_h_from_theta(theta_active: dict, theta_inactive: dict) -> tuple:
    """Delta_n_H(pH) = sum_i [theta_i,active(pH) - theta_i,inactive(pH)],
    computed directly from pre-computed per-site occupancy arrays rather
    than from a single per-site pKa.

    This is the form needed when theta_i(pH) comes from a coupled
    multi-site titration solve (``multisite.solve_titration``) rather than
    independent-site Henderson-Hasselbalch: a coupled site's theta(pH) is
    not, in general, expressible as a single effective pKa, so
    ``compute_linkage`` (which requires exactly that) cannot be used
    directly. The underlying identity is unchanged -- see the module
    docstring -- it only requires theta_i(pH) values, however derived.

    ``theta_active``/``theta_inactive`` map resnum -> theta(pH) array, all
    on the same pH grid; only resnums present in both are used, matching
    ``compute_linkage``'s convention for sites unresolved on one conformer.

    Returns ``(resnums, delta_n_h_per_residue, delta_n_h)`` with
    ``delta_n_h_per_residue`` shape (n_ph, n_sites) and ``delta_n_h`` shape
    (n_ph,) -- summed over sites, matching ``LinkageResult``'s fields.
    """
    resnums = sorted(set(theta_active) & set(theta_inactive))
    if not resnums:
        raise ValueError("no residues with theta(pH) in both theta_active and theta_inactive")
    per_residue = np.stack([theta_active[r] - theta_inactive[r] for r in resnums], axis=1)
    delta_n_h = np.nansum(per_residue, axis=1)
    return np.array(resnums), per_residue, delta_n_h


def delta_g_act_from_ln_z(ln_z_active, ln_z_inactive, T: float = 298.15):
    """DeltaDeltaG_act(pH) = -RT*[ln(Z_active) - ln(Z_inactive)], the
    coupled-titration generalization of ``compute_linkage``'s closed-form
    independent-site sum (which is exactly this expression specialized to
    Z = prod_i (1+10^(pKa_i-pH)), i.e. ln(Z) = sum_i log1p_10pow(pKa_i-pH)).

    Valid whenever ln(Z) is the properly shift-corrected total log
    partition function for each conformer's full titratable system -- e.g.
    ``multisite.MultiSiteTitrationResult.ln_z_total``, which sums
    independent clusters' ln(Z) (clusters multiply, so their logs add).
    """
    RT = R_KJ_PER_MOL_K * T
    ln_z_active = np.asarray(ln_z_active, dtype=float)
    ln_z_inactive = np.asarray(ln_z_inactive, dtype=float)
    return -RT * (ln_z_active - ln_z_inactive)


def sensitivity_band(results: list) -> dict:
    """Given several ``LinkageResult``s from perturbed inputs (e.g. with/
    without the Na+ ion, with/without rotamer relaxation -- pipeline spec
    step 4/step 1 sensitivity checks), return the min/max/spread of
    Delta_n_H(pH) across them at each pH, on a shared pH grid. Per the
    spec's guardrail: if the "headline" Delta_n_H magnitude is smaller
    than this spread, it must be reported as "not resolved," not a number
    (see ``LinkageResult`` callers / report generation, not enforced here).
    """
    ph = results[0].ph
    for r in results[1:]:
        if not np.array_equal(r.ph, ph):
            raise ValueError("all LinkageResults must share the same pH grid for a sensitivity band")
    stacked = np.array([r.delta_n_h for r in results])  # (n_variants, n_ph)
    return {
        "ph": ph,
        "min": stacked.min(axis=0),
        "max": stacked.max(axis=0),
        "spread": stacked.max(axis=0) - stacked.min(axis=0),
    }
