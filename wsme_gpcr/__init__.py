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
from .secondary_structure import (
    assign_secondary_structure,
    secondary_structure_from_codes,
    secondary_structure_from_dssp,
    DsspNotAvailableError,
)
from .contacts import ContactMap, compute_contact_map
from .blocking import BlockModel, build_blocks
from .wsme import WSMEParams, WSMEResult, run_wsme
from .dsc import compute_dsc, DSCResult
from .calibration import (
    CalibrationError,
    IsoStabilityResult,
    TmResult,
    XiTmCalibrationResult,
    XiFoldScanResult,
    PAPER_XI_MEAN_J_MOL,
    PAPER_XI_STD_J_MOL,
    PAPER_TARGET_TM_K,
    DEFAULT_FC_Z_THRESHOLD,
    DEFAULT_XI_SCAN_RANGE_J_MOL,
    DEFAULT_XI_SCAN_STEP_J_MOL,
    calibrate_xi_isostability_mode,
    calibrate_xi_tm_mode,
    compute_delta_g_fold,
    compute_fc,
    find_cp_peaks_and_tm,
    xi_fold_scan,
)
from .coupling import CouplingResult, compute_coupling
from .ion_pocket import IonPocketPartner, IonPocketResult, add_ion_pocket_interaction, place_na_ion
from .ionizable_network import IonizableNetworkResult, compute_ionizable_network, map_networks_to_blocks
from .alanine_scan import AlanineScanResult, run_alanine_scan, scannable_positions
from .pipeline import (
    PipelineResult,
    run_pipeline,
    run_pipeline_multi_ph,
    DEFAULT_PH_VALUES,
    AlanineScanPipelineResult,
    run_alanine_scan_pipeline,
)
from .asr import (
    NodePosteriors,
    AsrSensitivityResult,
    DEFAULT_POSTERIOR_THRESHOLD,
    DEFAULT_DELTA_TOLERANCE_FRAC,
    parse_iqtree_state_file,
    site_to_resnum,
    ambiguous_core_resnums,
    run_asr_sensitivity_check,
    evaluate_node_trustworthiness,
)

__all__ = [
    "Structure",
    "load_structure",
    "assign_secondary_structure",
    "secondary_structure_from_codes",
    "secondary_structure_from_dssp",
    "DsspNotAvailableError",
    "ContactMap",
    "compute_contact_map",
    "BlockModel",
    "build_blocks",
    "WSMEParams",
    "WSMEResult",
    "run_wsme",
    "compute_dsc",
    "DSCResult",
    "CalibrationError",
    "IsoStabilityResult",
    "TmResult",
    "XiTmCalibrationResult",
    "XiFoldScanResult",
    "PAPER_XI_MEAN_J_MOL",
    "PAPER_XI_STD_J_MOL",
    "PAPER_TARGET_TM_K",
    "DEFAULT_FC_Z_THRESHOLD",
    "DEFAULT_XI_SCAN_RANGE_J_MOL",
    "DEFAULT_XI_SCAN_STEP_J_MOL",
    "calibrate_xi_isostability_mode",
    "calibrate_xi_tm_mode",
    "compute_delta_g_fold",
    "xi_fold_scan",
    "compute_fc",
    "find_cp_peaks_and_tm",
    "CouplingResult",
    "compute_coupling",
    "IonPocketPartner",
    "IonPocketResult",
    "add_ion_pocket_interaction",
    "place_na_ion",
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
    "AlanineScanPipelineResult",
    "run_alanine_scan_pipeline",
    "NodePosteriors",
    "AsrSensitivityResult",
    "DEFAULT_POSTERIOR_THRESHOLD",
    "DEFAULT_DELTA_TOLERANCE_FRAC",
    "parse_iqtree_state_file",
    "site_to_resnum",
    "ambiguous_core_resnums",
    "run_asr_sensitivity_check",
    "evaluate_node_trustworthiness",
]
