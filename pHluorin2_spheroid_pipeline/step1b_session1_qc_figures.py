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

"""Step 1b (optional) - segmentation QC montage and the max-projection vs ratio-first projection pilot.
Outputs: segmentation_qc.svg, projection_method_comparison.svg (written to OUT_DIR/qc_session1)."""
os.makedirs('qc_session1', exist_ok=True); os.chdir('qc_session1')
run_lib('common.py')
RI24 = raw_index(PLATES['24h'])
exec(open(LIB('fig_session1_qc.py'), encoding='utf-8').read(), globals())
print(f'step 1b done in {time.time()-T0:.0f}s')
