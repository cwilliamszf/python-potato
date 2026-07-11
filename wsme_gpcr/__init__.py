"""wsme_gpcr: a Python port of the blocked WSME (bWSME) statistical mechanical
model for protein/GPCR conformational free-energy landscapes.

Ported from the MATLAB reference implementations:
  - https://github.com/AthiNaganathan/WSMEmodel
  - https://github.com/AthiNaganathan/GPCR-Landscapes

Reference:
  Gopi S, Aranganathan A, Naganathan AN. "Thermodynamics and folding
  landscapes of large proteins from a statistical mechanical model."
  Curr Res Struct Biol. 2019 Oct 23;1:6-12.
"""

from .structure import Structure, load_structure
from .secondary_structure import assign_secondary_structure
from .contacts import ContactMap, compute_contact_map
from .blocking import BlockModel, build_blocks
from .wsme import WSMEParams, WSMEResult, run_wsme
from .dsc import compute_dsc, DSCResult
from .coupling import CouplingResult, compute_coupling
from .ionizable_network import IonizableNetworkResult, compute_ionizable_network, map_networks_to_blocks
from .alanine_scan import AlanineScanResult, run_alanine_scan, scannable_positions
from .pipeline import PipelineResult, run_pipeline, run_pipeline_multi_ph, DEFAULT_PH_VALUES

__all__ = [
    "Structure",
    "load_structure",
    "assign_secondary_structure",
    "ContactMap",
    "compute_contact_map",
    "BlockModel",
    "build_blocks",
    "WSMEParams",
    "WSMEResult",
    "run_wsme",
    "compute_dsc",
    "DSCResult",
    "CouplingResult",
    "compute_coupling",
    "IonizableNetworkResult",
    "compute_ionizable_network",
    "map_networks_to_blocks",
    "AlanineScanResult",
    "run_alanine_scan",
    "scannable_positions",
    "PipelineResult",
    "run_pipeline",
    "run_pipeline_multi_ph",
    "DEFAULT_PH_VALUES",
]
