"""Multi-site titration solver: combines per-site intrinsic pKa's and
pairwise electrostatic couplings (from ``titration.py``) into pH-dependent
protonation occupancies theta_i(pH) that properly reflect site-site
coupling -- exact Boltzmann enumeration over coupled clusters, not
independent-site Henderson-Hasselbalch. This is the treatment the pipeline
spec calls out as required for the histidine-shell / buried-carboxylate
cluster.

Physics
-------
For a set of n titratable sites with intrinsic pKa_i (``compute_intrinsic_
pka``, computed holding every *other* site at its default reference
protonation state -- the Bashford-Karplus reduced-site approximation) and
pairwise coupling energies W_ij (kJ/mol, from ``compute_pairwise_coupling``
-- the excess energy present only when sites i and j are protonated
together, by construction zero when i and j do not interact), the
microstate energy for a protonation-state vector x in {0,1}^n (x_i=1 means
protonated) at a given pH is

    G(x, pH) = sum_i x_i * RT*ln10*(pH - pKa_i) + sum_{i<j} x_i*x_j*W_ij

The per-site energy term RT*ln10*(pH-pKa_i) is derived from requiring that
a single isolated site (n=1, no coupling) reproduce the standard
Henderson-Hasselbalch occupancy theta(pH) = 1/(1+10^(pH-pKa)) exactly under
two-state Boltzmann weighting with E(deprotonated)=0 -- verified directly
in ``tests/test_multisite.py`` against ``linkage.protonation_fraction``.

Sites are partitioned into connected components under a coupling-strength
threshold graph (``cluster_sites``); different components are statistically
independent (their joint microstate energy has no cross term), so each
cluster's 2^n microstates can be enumerated exactly and separately --
tractable because real coupling clusters (a histidine shell, a buried
carboxylate pair) are small, not because the whole titratable system is.

ln(Z) tracking
--------------
theta_i(pH) is normalization-invariant, so the numerically-necessary
per-pH energy shift (subtracting min(G) before exponentiating, to avoid
overflow) cancels exactly in theta. It does NOT cancel in ln(Z), which is
needed to generalize ``linkage.compute_linkage``'s closed-form
DeltaDeltaG_act(pH) to the coupled case (DeltaDeltaG_act(pH) =
-RT*[ln(Z_active)-ln(Z_inactive)]). ``solve_cluster_titration`` therefore
re-adds the shift (-g_min/RT) into the returned ln(Z), and cluster ln(Z)
values are additive across independent clusters (``ln_z_total`` in
``solve_titration``) -- see ``linkage.delta_g_act_from_ln_z``.

Beyond pairwise coupling: exact joint-microstate clusters
-----------------------------------------------------------
``solve_cluster_titration`` (above) builds G(x, pH) from per-site intrinsic
pKa's (each computed holding every *other* site frozen at a reference
state -- the Bashford-Karplus reduced-site approximation) plus pairwise
coupling corrections. That decomposition was found, in this pipeline's
development, to break down for a real tightly-interacting cluster (a 4-
residue GPCR loop, all within ~4-12 A of each other): individual intrinsic
pKa's shifted by >20 units from their model values, far beyond anything
documented in the literature, and the anomaly did not resolve with more
surrounding structural context or finer PB grids. ``solve_cluster_titration
_exact`` (below) sidesteps the decomposition for small clusters by
consuming *directly computed* whole-cluster joint microstate energies
(``titration.compute_cluster_joint_energies``) instead -- exact given those
energies, with no assumption that the electrostatics is additive over
sites plus pairwise terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .linkage import LN10, R_KJ_PER_MOL_K

MAX_EXACT_CLUSTER_SIZE = 20
DEFAULT_COUPLING_THRESHOLD_KJ_MOL = 2.5  # ~1 kT at 298 K (RT = 2.478 kJ/mol)


def _connected_components(nodes, edges) -> list:
    """Connected components of an undirected graph given as a node list
    and an iterable of adjacency pairs. Self-contained here rather than
    imported from ``wsme_gpcr.ionizable_network`` -- that module belongs
    to a separate tool (WSME) with its own independent implementation and
    lifecycle; ``linkage_pka`` does not depend on ``wsme_gpcr``."""
    adjacency = {n: set() for n in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = set()
    components = []
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(comp))
    return components


def cluster_sites(
    resnums,
    coupling: dict,
    coupling_threshold_kj_mol: float = DEFAULT_COUPLING_THRESHOLD_KJ_MOL,
) -> list:
    """Partition ``resnums`` into connected components under the coupling
    graph (an edge i-j exists iff |W_ij| >= threshold) -- sites in
    different components are statistically independent and can each be
    titrated as an isolated Henderson-Hasselbalch site (or as their own
    singleton "cluster", which ``solve_cluster_titration`` reproduces
    exactly; see module docstring).

    ``coupling`` maps (resnum_i, resnum_j) -> W_ij in kJ/mol; either key
    order is accepted. Pairs not present in ``coupling`` are treated as
    uncoupled (W_ij=0), matching ``compute_pairwise_coupling``'s implicit
    assumption that only interacting pairs are computed and stored.
    """
    resnums = list(resnums)
    edges = []
    for (i, j), w in coupling.items():
        if i in resnums and j in resnums and abs(w) >= coupling_threshold_kj_mol:
            edges.append((i, j))
    return _connected_components(resnums, edges)


def _lookup_coupling(coupling: dict, i, j) -> float:
    if (i, j) in coupling:
        return coupling[(i, j)]
    if (j, i) in coupling:
        return coupling[(j, i)]
    return 0.0


@dataclass
class ClusterTitrationResult:
    resnums: list        # sites in this cluster, in the order used for theta/state axes
    ph: np.ndarray        # (n_ph,)
    theta: dict            # resnum -> (n_ph,) fraction-protonated array
    ln_z: np.ndarray       # (n_ph,) properly shift-corrected ln(partition function)


def solve_cluster_titration(
    pka_intrinsic: dict,
    coupling: dict,
    cluster_resnums: list,
    ph_values,
    temp_k: float = 298.15,
) -> ClusterTitrationResult:
    """Exact Boltzmann enumeration of all 2^n protonation microstates for
    one coupled cluster of sites, at every pH in ``ph_values``. Vectorized
    over pH and microstates simultaneously.

    Raises ``ValueError`` for clusters exceeding ``MAX_EXACT_CLUSTER_SIZE``
    rather than silently falling back to a mean-field or sampling
    approximation -- per the pipeline's "no MD/conformational sampling,
    exact given the pKa's" mandate, an oversized coupled cluster is a
    genuine limitation to surface, not to paper over.
    """
    n = len(cluster_resnums)
    if n == 0:
        raise ValueError("cluster_resnums must be non-empty")
    if n > MAX_EXACT_CLUSTER_SIZE:
        raise ValueError(
            f"cluster of {n} coupled sites ({cluster_resnums}) exceeds "
            f"MAX_EXACT_CLUSTER_SIZE={MAX_EXACT_CLUSTER_SIZE} (2^{n} "
            "microstates is not tractable for exact enumeration); no "
            "approximate fallback is implemented here -- only raise "
            "coupling_threshold_kj_mol if physically justified, do not "
            "silently truncate the cluster."
        )

    RT = R_KJ_PER_MOL_K * temp_k
    ph = np.asarray(list(ph_values), dtype=float)
    pka = np.array([pka_intrinsic[r] for r in cluster_resnums], dtype=float)  # (n,)

    n_states = 2 ** n
    # (n_states, n) occupancy matrix: states[s, i] = bit i of state index s.
    states = ((np.arange(n_states)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)

    # Symmetric pairwise coupling matrix, zero diagonal.
    W = np.zeros((n, n))
    for a, b in combinations(range(n), 2):
        w_ab = _lookup_coupling(coupling, cluster_resnums[a], cluster_resnums[b])
        W[a, b] = w_ab
        W[b, a] = w_ab

    # Per-site protonated-state energy RT*ln10*(pH-pKa_i), shape (n_ph, n).
    linear = RT * LN10 * (ph[:, None] - pka[None, :])
    # sum_i x_i * linear_i for every (pH, state): (n_ph, n) @ (n, n_states).
    g_linear = linear @ states.T  # (n_ph, n_states)

    # sum_{i<j} x_i*x_j*W_ij per state (pH-independent): (n_states,).
    g_pairwise = 0.5 * np.einsum("si,ij,sj->s", states, W, states)

    G = g_linear + g_pairwise[None, :]  # (n_ph, n_states) kJ/mol

    g_min = G.min(axis=1, keepdims=True)      # (n_ph, 1) per-pH numerical shift
    weights = np.exp(-(G - g_min) / RT)        # (n_ph, n_states)
    z_shifted = weights.sum(axis=1)             # (n_ph,)

    # Re-add the shift so ln_z is the true, additive-across-clusters log
    # partition function -- not just the shifted value (see module docstring).
    ln_z = np.log(z_shifted) - g_min[:, 0] / RT  # (n_ph,)

    theta = {}
    for idx, resnum in enumerate(cluster_resnums):
        occ = states[:, idx]  # (n_states,)
        theta[resnum] = (weights @ occ) / z_shifted  # (n_ph,)

    return ClusterTitrationResult(resnums=list(cluster_resnums), ph=ph, theta=theta, ln_z=ln_z)


def solve_cluster_titration_exact(
    sites: list,
    joint_energies: dict,
    model_pka: dict,
    dg_ion_model: dict,
    ph_values,
    temp_k: float = 298.15,
) -> ClusterTitrationResult:
    """Exact multi-site titration for one tightly-coupled cluster using
    DIRECTLY computed whole-system joint microstate energies
    (``titration.compute_cluster_joint_energies``), rather than
    ``solve_cluster_titration``'s intrinsic-pKa + pairwise-coupling
    decomposition. This avoids the Bashford-Karplus reduced-site
    approximation (freezing every *other* cluster member at a fixed
    reference state while computing each site's own intrinsic pKa) --
    which was found, in this pipeline's development, to break down for a
    real tightly-interacting GPCR loop cluster (see
    ``compute_cluster_joint_energies``'s docstring for the numbers).

    Derivation: for an isolated single site (n=1), matching
    ``linkage.protonation_fraction``'s convention, the protonated state
    (x=1) energy relative to deprotonated (x=0) must equal
    ``RT*ln10*(pH - intrinsic_pka)``, where (per
    ``titration.compute_intrinsic_pka``) ``intrinsic_pka = model_pka +
    (dG_ion_protein - dG_ion_model)/(RT ln10)`` and ``dG_ion_protein =
    E_protein(deprot) - E_protein(prot) = -(E_protein(x=1)-E_protein(x=0))``.
    Substituting and generalizing to a joint microstate x (sum over sites,
    plus the *directly computed* whole-cluster protein energy difference
    in place of a per-site decomposition) gives

        G(x, pH) = sum_i x_i*[RT*ln10*(pH - model_pka_i) + dG_ion_model_i]
                   + [E_protein(x) - E_protein(x=all-deprotonated)]

    which reduces exactly to ``solve_cluster_titration``'s single-site
    result, without assuming ``E_protein(x)`` decomposes into per-site plus
    pairwise terms (it needn't, which is the whole point of computing it
    directly for small, tightly-coupled clusters).

    ``sites``: ``[(resnum, resname), ...]`` in the same order used to build
    ``joint_energies``' occupancy-tuple keys.
    ``joint_energies``: ``{occupancy_tuple: E_protein(x) kJ/mol}`` from
    ``compute_cluster_joint_energies`` -- must include the
    all-``False`` (fully deprotonated) key as the reference state.
    ``model_pka``, ``dg_ion_model``: resnum -> float, the model-compound
    terms already computed per-site (e.g. from ``SiteEnergyResult``),
    unaffected by cluster context.
    """
    resnums = [r for r, _ in sites]
    n = len(sites)
    if n > MAX_EXACT_CLUSTER_SIZE:
        raise ValueError(
            f"cluster of {n} sites exceeds MAX_EXACT_CLUSTER_SIZE={MAX_EXACT_CLUSTER_SIZE} "
            f"(2^{n} joint microstates is not tractable for exact enumeration)"
        )

    RT = R_KJ_PER_MOL_K * temp_k
    ph = np.asarray(list(ph_values), dtype=float)

    n_states = 2 ** n
    states = ((np.arange(n_states)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)  # (n_states, n)

    all_deprot = tuple(False for _ in range(n))
    if all_deprot not in joint_energies:
        raise ValueError("joint_energies must include the all-deprotonated reference state")
    e_ref = joint_energies[all_deprot]

    e_state = np.array([joint_energies[tuple(bool(b) for b in row)] for row in states]) - e_ref  # (n_states,)

    site_model_pka = np.array([model_pka[r] for r in resnums])   # (n,)
    site_dg_model = np.array([dg_ion_model[r] for r in resnums])  # (n,)

    states_f = states.astype(float)
    linear_per_site = RT * LN10 * (ph[:, None] - site_model_pka[None, :])  # (n_ph, n)
    g_linear = linear_per_site @ states_f.T                                 # (n_ph, n_states)
    g_const_model = states_f @ site_dg_model                                # (n_states,)

    G = g_linear + (g_const_model + e_state)[None, :]  # (n_ph, n_states) kJ/mol

    g_min = G.min(axis=1, keepdims=True)
    weights = np.exp(-(G - g_min) / RT)
    z_shifted = weights.sum(axis=1)
    ln_z = np.log(z_shifted) - g_min[:, 0] / RT

    theta = {}
    for idx, resnum in enumerate(resnums):
        occ = states_f[:, idx]
        theta[resnum] = (weights @ occ) / z_shifted

    return ClusterTitrationResult(resnums=resnums, ph=ph, theta=theta, ln_z=ln_z)


@dataclass
class MultiSiteTitrationResult:
    ph: np.ndarray
    theta: dict            # resnum -> (n_ph,) array, covering every input site
    clusters: list          # list of lists of resnums (the coupling partition)
    cluster_results: list    # list of ClusterTitrationResult, aligned to clusters
    ln_z_total: np.ndarray   # (n_ph,) sum of per-cluster ln(Z)


def solve_titration(
    pka_intrinsic: dict,
    coupling: dict,
    ph_values,
    coupling_threshold_kj_mol: float = DEFAULT_COUPLING_THRESHOLD_KJ_MOL,
    temp_k: float = 298.15,
) -> MultiSiteTitrationResult:
    """Top-level multi-site titration: cluster sites by coupling strength,
    exactly enumerate each cluster's Boltzmann distribution, and combine
    into per-site theta(pH) covering every site in ``pka_intrinsic``.

    Clusters are independent by construction (no coupling term links sites
    in different clusters), so their partition functions multiply and
    ln(Z_total) = sum of per-cluster ln(Z) -- pass ``ln_z_total`` for the
    active and inactive conformers into ``linkage.delta_g_act_from_ln_z``
    to get DeltaDeltaG_act(pH) under coupling, generalizing
    ``linkage.compute_linkage``'s independent-site closed form.
    """
    resnums = sorted(pka_intrinsic)
    clusters = cluster_sites(resnums, coupling, coupling_threshold_kj_mol)

    ph = np.asarray(list(ph_values), dtype=float)
    theta = {}
    cluster_results = []
    ln_z_total = np.zeros_like(ph)

    for cluster_resnums in clusters:
        result = solve_cluster_titration(pka_intrinsic, coupling, cluster_resnums, ph, temp_k=temp_k)
        cluster_results.append(result)
        theta.update(result.theta)
        ln_z_total = ln_z_total + result.ln_z

    return MultiSiteTitrationResult(
        ph=ph,
        theta=theta,
        clusters=clusters,
        cluster_results=cluster_results,
        ln_z_total=ln_z_total,
    )
