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
]
