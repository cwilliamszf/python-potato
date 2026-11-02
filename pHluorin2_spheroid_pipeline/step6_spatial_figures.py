import os, sys, time, gc, json, pickle, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'lib'))
from config import *
from figstyle import apply_figure_style, panel_letter, goodness_arrow
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import scipy.ndimage as ndi, tifffile
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
mpl.rcParams['svg.fonttype'] = 'none'


def LIB(name):
    return os.path.join(HERE, 'lib', name)


def run_lib(name):
    exec(open(LIB(name), encoding='utf-8').read(), globals())


LATE = TPS_ALL[-2:]; LAST = TPS_ALL[-1]; LASTPAIR = f'{TPS_ALL[-2]}->{TPS_ALL[-1]}'   # original: 48h,72h / 72h / 48h->72h
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
T0 = time.time()
display_order = DISPLAY_ORDER_CFG or (['DMSO', 'Fluorobrite', '1uM TMZ'] + [f'{d} {s}' for d in ['Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace'] for s in 'LMH'] + \
    ['0.5uM OGM', '1uM OGM', '5uM OGM', '20uM OGM', '0.5uM Ogerin', '5uM Ogerin', '20uM Ogerin', '50uM Ogerin'])

"""Step 6 - spatial figures: per-compound 488/405 + patch maps (maps_<compound>.svg), dose x time grid
(dose_time_spatial_grid.svg), tracking / matched-mean / controls / trajectories (robustness_tracking_trajectories.svg)."""
run_lib('common.py'); run_lib('stage2_features.py'); run_lib('analysis_stats.py')
Fv = pd.read_csv('spatial_features_v2.csv')
MAPS = {k: {'_cal': v} for k, v in pickle.load(open('spatial_maps_cal.pkl', 'rb')).items()}
COMPOUNDS = COMPOUNDS_CFG or ['Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace', 'OGM', 'Ogerin']
LOWSIG = set(zip(Fv[Fv.low_signal].timepoint, Fv[Fv.low_signal].well))
run_lib('fig_compound_maps.py')
for cpd in (MAP_SERIES_CFG or ['Controls'] + COMPOUNDS):
    compound_figure(cpd, MAPS, Fv, '_cal', (0.9, 2.4), reslim=0.15, fname=f'maps_{cpd}.svg')
FEAT = {'ratio_median_cal': 'mean acidity (median 488/405)', 'rim_minus_core_cal': 'rim − centre 488/405',
        'resid_sd_rob_cal': 'residual SD (robust, detrended)', 'acid_patch_frac_cal': 'acidic-patch area fraction',
        'corr_len_rob_cal': 'correlation length (full-res px)', 'patch_boundary_enrich_cal': 'patch boundary enrichment (outer 20 %)',
        'area_px': 'spheroid area (full-res px²)'}
good = Fv[~Fv.qc_exclude]; gq = good[good.condition != 'Fluorobrite'].copy()
EX = pd.read_csv('radial_excess_profiles.csv')
TS = pd.read_csv('tracking_same_spheroid.csv'); TN = pd.read_csv('tracking_null_pairs.csv')
MM = {h: (pd.read_csv(f'matched_mean_{h}.csv'), None) for h in ['resid_sd_rob_cal', 'acid_patch_frac_cal', 'corr_len_rob_cal']}
PF = pd.read_csv('matched_mean_family_tests.csv'); ADJ = pd.read_csv('covariate_adjusted_effects.csv'); LOO = pd.read_csv('trajectory_classification.csv')
print(len(gq), len(EX), len(TS), len(TN), [len(v[0]) for v in MM.values()], len(PF), len(ADJ), len(LOO), LOO.label.unique())

run_lib('fig_dose_time.py')
run_lib('fig_robustness.py')
print(f'step 6 done in {time.time()-T0:.0f}s')
