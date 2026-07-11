"""linkage_pka: Wyman/Tanford proton-linkage analysis of GPCR activation --
does acidification thermodynamically favor the active state, and by how
many protons -- from Poisson-Boltzmann pKa's on fixed active/inactive
structures. No molecular dynamics, no conformational sampling.

This is a deliberately separate tool from ``wsme_gpcr``: WSME normalizes
each pH ensemble's partition function to 1 internally, which divides out
exactly the inter-pH free-energy offset this calculation needs. See
``linkage.py`` for the physics.
"""

from .linkage import (
    LinkageResult,
    compute_linkage,
    protonation_fraction,
    sensitivity_band,
)
from .structure_prep import (
    CHI_ATOMS,
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
    GridParams,
    SiteEnergyResult,
    TITRATABLE_RESIDUES,
    PqrAtom,
    build_microstate,
    build_model_compound_atoms,
    charge_delta,
    compute_environment_energies,
    compute_intrinsic_pka,
    compute_pairwise_coupling,
    compute_solvation_energy,
    load_amber_charges,
    place_titratable_hydrogen,
    read_pqr,
    write_pqr,
)

__all__ = [
    "LinkageResult",
    "compute_linkage",
    "protonation_fraction",
    "sensitivity_band",
    "CHI_ATOMS",
    "IONIZABLE_RESNAMES",
    "PrepResult",
    "RotamerChoice",
    "measure_chi",
    "optimize_rotamers",
    "run_structure_prep",
    "MembraneFrame",
    "compute_membrane_frame",
    "find_r350",
    "find_y753",
    "DxMap",
    "compute_dummy_maps",
    "compute_energy_with_maps",
    "read_dx",
    "splice_membrane_slab",
    "write_dx",
    "write_maps",
    "GridParams",
    "SiteEnergyResult",
    "TITRATABLE_RESIDUES",
    "PqrAtom",
    "build_microstate",
    "build_model_compound_atoms",
    "charge_delta",
    "compute_environment_energies",
    "compute_intrinsic_pka",
    "compute_pairwise_coupling",
    "compute_solvation_energy",
    "load_amber_charges",
    "place_titratable_hydrogen",
    "read_pqr",
    "write_pqr",
]
