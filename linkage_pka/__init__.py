"""linkage_pka: Wyman/Tanford proton-linkage analysis of GPCR activation --
does acidification thermodynamically favor the active state, and by how
many protons -- from Poisson-Boltzmann pKa's on fixed active/inactive
structures. No molecular dynamics (no trajectories, no time integration).
Conformational ensemble sampling (multi-conformer/rotamer-library scoring,
MCCE-style) is in scope and, per Gate A (see ``gate_a.py``,
``FINDINGS.md``), necessary: single-structure PB with at most one relaxed
rotamer per microstate (``titration.optimize_rotamer_for_microstate``)
fails Gate A's real-experimental-data acceptance test by 8-11x, so an
ensemble/multi-conformer extension is the next real step, not yet built.
Every relaxation variant is always reported alongside the rigid-geometry
result, never silently substituted for it.

This is a deliberately separate tool from ``wsme_gpcr``: WSME normalizes
each pH ensemble's partition function to 1 internally, which divides out
exactly the inter-pH free-energy offset this calculation needs. See
``linkage.py`` for the physics.
"""

from .linkage import (
    LinkageResult,
    compute_linkage,
    delta_g_act_from_ln_z,
    delta_n_h_from_theta,
    protonation_fraction,
    sensitivity_band,
)
from .structure_prep import (
    CHI_ATOMS,
    EXTRA_CHI_ATOMS,
    IONIZABLE_RESNAMES,
    PrepResult,
    RotamerChoice,
    measure_chi,
    optimize_rotamers,
    run_structure_prep,
)
from .membrane_frame import (
    MembraneFrame,
    compute_membrane_frame,
    find_d250,
    find_r350,
    find_y753,
)
from .dielectric_map import (
    DxMap,
    compute_dummy_maps,
    compute_energy_with_maps,
    read_dx,
    splice_membrane_slab,
    write_dx,
    write_maps,
)
from .titration import (
    ALL_CHI_ATOMS,
    COULOMB_CONSTANT_KJ_ANG_PER_MOL_E2,
    GridParams,
    SiteEnergyResult,
    TITRATABLE_RESIDUES,
    PqrAtom,
    build_microstate,
    build_model_compound_atoms,
    build_na_ion_atom,
    charge_delta,
    compute_environment_energies,
    compute_intrinsic_pka,
    compute_cluster_joint_energies,
    compute_pairwise_coupling,
    compute_solvation_energy,
    find_relaxation_neighbors,
    load_amber_charges,
    load_na_ion_parameters,
    optimize_rotamer_for_microstate,
    optimize_rotamers_with_neighbors,
    place_na_ion,
    place_titratable_hydrogen,
    read_pqr,
    write_pqr,
)
from .multisite import (
    MAX_EXACT_CLUSTER_SIZE,
    DEFAULT_COUPLING_THRESHOLD_KJ_MOL,
    ClusterTitrationResult,
    MultiSiteTitrationResult,
    cluster_sites,
    solve_cluster_titration,
    solve_cluster_titration_exact,
    solve_titration,
)
from .pipeline import (
    ActivationLinkageResult,
    StructureTitrationResult,
    compute_activation_linkage,
    find_coupled_pairs,
    identify_titratable_sites,
    residue_min_distance,
    run_structure_titration,
)
from .gate_a import (
    GATE_A_RMSE_THRESHOLD_PKA_UNITS,
    REFERENCE_PDB_ID,
    SNASE_1STN_EXPERIMENTAL_PKA,
    ExperimentalPka,
    GateAResult,
    compute_gate_a_rmse,
)

__all__ = [
    "LinkageResult",
    "compute_linkage",
    "delta_g_act_from_ln_z",
    "delta_n_h_from_theta",
    "protonation_fraction",
    "sensitivity_band",
    "CHI_ATOMS",
    "EXTRA_CHI_ATOMS",
    "IONIZABLE_RESNAMES",
    "PrepResult",
    "RotamerChoice",
    "measure_chi",
    "optimize_rotamers",
    "run_structure_prep",
    "MembraneFrame",
    "compute_membrane_frame",
    "find_d250",
    "find_r350",
    "find_y753",
    "DxMap",
    "compute_dummy_maps",
    "compute_energy_with_maps",
    "read_dx",
    "splice_membrane_slab",
    "write_dx",
    "write_maps",
    "ALL_CHI_ATOMS",
    "COULOMB_CONSTANT_KJ_ANG_PER_MOL_E2",
    "GridParams",
    "SiteEnergyResult",
    "TITRATABLE_RESIDUES",
    "PqrAtom",
    "build_microstate",
    "build_model_compound_atoms",
    "build_na_ion_atom",
    "charge_delta",
    "compute_environment_energies",
    "compute_intrinsic_pka",
    "compute_cluster_joint_energies",
    "compute_pairwise_coupling",
    "compute_solvation_energy",
    "find_relaxation_neighbors",
    "load_amber_charges",
    "load_na_ion_parameters",
    "optimize_rotamer_for_microstate",
    "optimize_rotamers_with_neighbors",
    "place_na_ion",
    "place_titratable_hydrogen",
    "read_pqr",
    "write_pqr",
    "MAX_EXACT_CLUSTER_SIZE",
    "DEFAULT_COUPLING_THRESHOLD_KJ_MOL",
    "ClusterTitrationResult",
    "MultiSiteTitrationResult",
    "cluster_sites",
    "solve_cluster_titration",
    "solve_cluster_titration_exact",
    "solve_titration",
    "ActivationLinkageResult",
    "StructureTitrationResult",
    "compute_activation_linkage",
    "find_coupled_pairs",
    "identify_titratable_sites",
    "residue_min_distance",
    "run_structure_titration",
    "GATE_A_RMSE_THRESHOLD_PKA_UNITS",
    "REFERENCE_PDB_ID",
    "SNASE_1STN_EXPERIMENTAL_PKA",
    "ExperimentalPka",
    "GateAResult",
    "compute_gate_a_rmse",
]
