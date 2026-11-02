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

"""Step 2 (legacy, superseded) - first spatial pass on raw (non-background-calibrated) 488/405 maps.
Reproduces spatial_overview.svg, spatial_maps_48h.svg, spatial_feature_heatmap.svg, ratio_flatfield.svg,
shared_residual_pattern.svg, aceM_bf_check.svg, spatial_features_per_well.csv, spatial_effect_sizes_vs_DMSO.csv.
SUPERSEDED by steps 4-7 (calibrated background; the DMSO 'acidic rim' seen here was a max-projection artefact).
This is a verbatim replay of the original interactive analysis cells; outputs go to OUT_DIR/legacy_spatial_v1."""
import shutil
os.makedirs('legacy_spatial_v1', exist_ok=True); os.chdir('legacy_spatial_v1')
shutil.copy(LIB('common.py'), 'spatial_pass1.py'); shutil.copy(LIB('spatial_pass2.py'), 'spatial_pass2.py')
shutil.copy(os.path.join(HERE, 'legacy', 'spatial_overview_fig.py'), 'spatial_overview_fig.py')
src = open(os.path.join(HERE, 'legacy', 'replay_spatial_v1.py'), encoding='utf-8').read()
exec(src, globals())
print(f'step 2 done in {time.time()-T0:.0f}s')
