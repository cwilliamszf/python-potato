"""Configuration for the MCF7 pHluorin2-GPI spheroid analysis pipeline.

Edit DATA_DIR (or set the environment variable SPHEROID_DATA_DIR) to point at the
Cytation export folder that contains the three plate sub-folders.
Readout convention throughout: 488/405 ratio, HIGHER = MORE ACIDIC.
"""
import os

import json

DATA_DIR = os.environ.get('SPHEROID_DATA_DIR',
                          r'C:\path\to\data\261001_105300_10.1.26 MCF7 P6 Spheroid Drug Assay')

# plate sub-folder for each imaging timepoint (same spheroids re-imaged), in time order
PLATE_FOLDERS = {'24h': '261001_105300_Plate 1',
                 '48h': '261002_173508_Plate 3',
                 '72h': '261003_211651_Plate 1'}
EXPERIMENT = 'MCF7_exp1'

# all outputs (tables, figures, intermediate arrays) are written here
OUT_DIR = os.environ.get('SPHEROID_OUT_DIR',
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))


def zproj_dir(tp):
    """Cytation max-projection sub-folder (name contains 'ZProj') of a plate."""
    p = PLATE_DIR[tp]
    hits = [d for d in os.listdir(p) if 'ZProj' in d and os.path.isdir(os.path.join(p, d))]
    if not hits:
        raise FileNotFoundError(f'no *ZProj* folder in {p}')
    return os.path.join(p, hits[0])

# The plate layout (condition per well) is defined in lib/common.py (`layout`),
# the L/M/H doses in lib/stage2_features.py (`DOSE_UM`), and the QC exclusions
# below. Change them there if you adapt the pipeline to another plate.

# wells excluded at every timepoint (Ace M H5: fibre across spheroid; H6: debris shadow)
EXCLUDE_WELLS_ALL_TP = ['H5', 'H6']
# spheroid-timepoints with visible debris, excluded from statistics (qc_exclude)
VIS_DEBRIS = [('48h', 'H3'), ('72h', 'H3'), ('24h', 'B6'), ('72h', 'D9')]

# ---- other experiments: point SPHEROID_EXPERIMENT at a JSON file (see experiments/*.json) that overrides
# data_dir, plate_folders (ordered {timepoint: folder}), out_dir, exclude_wells_all_tp, vis_debris, name.
_EXP = os.environ.get('SPHEROID_EXPERIMENT')
if _EXP:
    _EXP = os.environ['SPHEROID_EXPERIMENT'] = os.path.abspath(_EXP)   # worker processes start in OUT_DIR
    _e = json.load(open(_EXP, encoding='utf-8'))
    EXPERIMENT = _e.get('name', os.path.splitext(os.path.basename(_EXP))[0])
    DATA_DIR = os.environ.get('SPHEROID_DATA_DIR', _e['data_dir'])
    PLATE_FOLDERS = dict(_e['plate_folders'])
    OUT_DIR = os.environ.get('SPHEROID_OUT_DIR', _e.get('out_dir', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_' + EXPERIMENT)))
    EXCLUDE_WELLS_ALL_TP = _e.get('exclude_wells_all_tp', [])
    VIS_DEBRIS = [tuple(x) for x in _e.get('vis_debris', [])]
    SEGMENTATION_CFG = _e.get('segmentation', 'v1')
    BF_CHANNEL = _e.get('bf_channel', 'Bright Field')
    LAYOUT_WELLS = _e.get('layout_wells')
    COMPOUNDS_CFG = _e.get('compounds')
    DISPLAY_ORDER_CFG = _e.get('display_order')
    CTRL_CONDS_CFG = _e.get('ctrl_conditions')
    NULL_CONDS_CFG = _e.get('null_conditions')
    COND_INFO_CFG = _e.get('cond_info')
    SERIES_CFG = _e.get('series')
    FULLNAME_CFG = _e.get('fullname')
    HUE_CFG = _e.get('hue')
    PERM_FAMILY_CFG = _e.get('perm_family_compounds')
    FIG_COMPOUNDS_CFG = _e.get('fig_compounds')
    ALPHA_FORCE = _e.get('alpha_force', False)
    SHAPE_QC_CFG = _e.get('shape_qc')
    MAP_SERIES_CFG = _e.get('map_series')
    ALPHA_FIXED = _e.get('alpha_fixed')
    LOW_SIGNAL_SBR = _e.get('low_signal_sbr', 1.03)

ALPHA_FIXED = globals().get('ALPHA_FIXED')   # None for the first experiment (alpha always fitted)
# per-experiment plate/condition overrides (None -> the original experiment-1 hard-coded values)
BF_CHANNEL = globals().get('BF_CHANNEL', 'Bright Field')   # Cytation transmitted-light channel name
LAYOUT_WELLS = globals().get('LAYOUT_WELLS')               # {row: [12 condition names]}
COMPOUNDS_CFG = globals().get('COMPOUNDS_CFG')
DISPLAY_ORDER_CFG = globals().get('DISPLAY_ORDER_CFG')
CTRL_CONDS_CFG = globals().get('CTRL_CONDS_CFG')           # conditions used for the background-alpha scan
NULL_CONDS_CFG = globals().get('NULL_CONDS_CFG')           # near-null conditions for the optical controls
COND_INFO_CFG = globals().get('COND_INFO_CFG')             # condition -> [compound, dose_um|null, dose_rank]
SERIES_CFG = globals().get('SERIES_CFG')                   # compound -> [[condition, label], ...]
FULLNAME_CFG = globals().get('FULLNAME_CFG')
HUE_CFG = globals().get('HUE_CFG')
PERM_FAMILY_CFG = globals().get('PERM_FAMILY_CFG')
ALPHA_FORCE = bool(globals().get('ALPHA_FORCE', False))    # always use alpha_fixed (per-timepoint fits unstable)
SHAPE_QC_CFG = globals().get('SHAPE_QC_CFG')               # roundness gate: {circularity_min, solidity_min, edge_roughness_max, z_ref_condition, exclude}
FIG_COMPOUNDS_CFG = globals().get('FIG_COMPOUNDS_CFG')     # columns of the dose x time grid
MAP_SERIES_CFG = globals().get('MAP_SERIES_CFG')           # which per-compound map figures step 6 draws
ALPHA_MIN_SPAN = 0.15
# spheroid flagged low_signal when 488 or 405 signal / medium < LOW_SIGNAL_SBR (1.03 in the original analysis)
LOW_SIGNAL_SBR = globals().get('LOW_SIGNAL_SBR', 1.03)
PLATE_DIR = {tp: os.path.join(DATA_DIR, f) for tp, f in PLATE_FOLDERS.items()}
TPS_ALL = list(PLATE_FOLDERS)                      # timepoints in acquisition order
HOURS = {tp: float(tp.rstrip('h')) for tp in TPS_ALL}
