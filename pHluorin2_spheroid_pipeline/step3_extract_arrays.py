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

"""Step 3 - single I/O pass: segment every spheroid on bright field and store 4x4-binned (490x490 px; 1 px = 4
full-res px) 488 and 405 max-projection intensities plus masks. Outputs: stage1_arrays.npz, stage1_status.csv."""
run_lib('stage1_extract.py')
G, Rr, M, s1 = run_stage1()
np.savez_compressed('stage1_arrays.npz',
    keys=np.array([f'{t}|{w}' for t, w in G]),
    g=np.stack([G[k] for k in G]), r=np.stack([Rr[k] for k in G]),
    m=np.stack([M.get(k, np.zeros((490, 490), bool)) for k in G]))
s1.to_csv('stage1_status.csv', index=False)
ALT = os.environ.get('SPHEROID_ALT_OUT')   # v2 loader: also write the equatorial-plane arrays (same masks) here
if GP and ALT:
    os.makedirs(ALT, exist_ok=True)
    np.savez_compressed(os.path.join(ALT, 'stage1_arrays.npz'), keys=np.array([f'{t}|{w}' for t, w in G]),
        g=np.stack([GP[k] for k in G]), r=np.stack([RP[k] for k in G]),
        m=np.stack([M.get(k, np.zeros((490, 490), bool)) for k in G]))
    s1.to_csv(os.path.join(ALT, 'stage1_status.csv'), index=False)
    print('equatorial-plane arrays written to', ALT)
print(s1.status.value_counts().to_dict(), 'debris-trimmed masks:', int(s1.mask_split.sum()))
print(f'step 3 done in {time.time()-T0:.0f}s')
