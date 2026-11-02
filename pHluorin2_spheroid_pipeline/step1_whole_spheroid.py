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

"""Step 1 - whole-spheroid 488/405 measurements and summary figures (first analysis).
Inputs : raw Cytation folders (DATA_DIR). Outputs: file_inventory.csv, per_well_measurements.csv,
condition_summary.csv, dose_response_params.csv, plate_heatmaps/condition_heatmap/timecourse_plot/bar_charts/
dose_response .svg, representative_images.svg, representative_ratio_images.svg.
Wells without a Cytation ZProj image get a max-intensity projection built from their raw z-slices."""
from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
print('== inventory', flush=True)

import os, re
import pandas as pd
import numpy as np

BASE = DATA_DIR

# ── Plate layout ────────────────────────────────────────────────────────────
rows = list('ABCDEFGH')
cols = list(range(1, 13))

raw_layout = {
    'A': {(1,3):'DMSO',      (4,6):'Fluorobrite',  (7,9):'1uM TMZ',    (10,12):'0.5uM OGM'},
    'B': {(1,3):'Oz L',      (4,6):'Oz M',          (7,9):'Oz H',        (10,12):'1uM OGM'},
    'C': {(1,3):'Baf L',     (4,6):'Baf M',         (7,9):'Baf H',       (10,12):'5uM OGM'},
    'D': {(1,3):'Rot L',     (4,6):'Rot M',         (7,9):'Rot H',       (10,12):'20uM OGM'},
    'E': {(1,3):'3BP L',     (4,6):'3BP M',         (7,9):'3BP H',       (10,12):'0.5uM Ogerin'},
    'F': {(1,3):'Phlor L',   (4,6):'Phlor M',       (7,9):'Phlor H',     (10,12):'5uM Ogerin'},
    'G': {(1,3):'Olig L',    (4,6):'Olig M',        (7,9):'Olig H',      (10,12):'20uM Ogerin'},
    'H': {(1,3):'Ace L',     (4,6):'Ace M',         (7,9):'Ace H',       (10,12):'50uM Ogerin'},
}

plate_map = {}
for row_letter, blocks in raw_layout.items():
    for (c_lo, c_hi), condition in blocks.items():
        for c in range(c_lo, c_hi + 1):
            plate_map[f"{row_letter}{c}"] = condition

# ── Plate directories & timepoints ──────────────────────────────────────────
plate_dirs = dict(PLATE_DIR)

def find_zproj_dir(plate_path):
    for d in os.listdir(plate_path):
        if 'ZProj' in d:
            return os.path.join(plate_path, d)
    return None

def index_zproj(zproj_dir):
    """Return dict: well -> {gfp, rogfp, bf}"""
    files = os.listdir(zproj_dir)
    index = {}
    for f in files:
        if not f.endswith('.tif'):
            continue
        well = f.split('_')[0]
        fp = os.path.join(zproj_dir, f)
        if 'GFP 469' in f:
            index.setdefault(well, {})['gfp'] = fp
        elif 'Oxidized roGFP2' in f:
            index.setdefault(well, {})['rogfp'] = fp
        elif 'Bright Field' in f:
            index.setdefault(well, {})['bf'] = fp
    return index

# ── Build inventory ──────────────────────────────────────────────────────────
records = []
all_wells = [f"{r}{c}" for r in rows for c in cols]

for tp, plate_path in plate_dirs.items():
    zproj_dir = find_zproj_dir(plate_path)
    zproj_index = index_zproj(zproj_dir) if zproj_dir else {}
    
    for well in all_wells:
        condition = plate_map.get(well, 'Unknown')
        entry = zproj_index.get(well, {})
        records.append({
            'timepoint': tp,
            'plate_dir': plate_path,
            'zproj_dir': zproj_dir,
            'well': well,
            'condition': condition,
            'gfp_file': entry.get('gfp', ''),
            'rogfp_file': entry.get('rogfp', ''),
            'bf_file': entry.get('bf', ''),
            'zproj_exists': bool(entry.get('gfp') and entry.get('rogfp') and entry.get('bf')),
        })

inv = pd.DataFrame(records)
inv.to_csv('file_inventory.csv', index=False)

# Summary
total = len(inv)
has_zproj = inv['zproj_exists'].sum()
missing   = total - has_zproj

print(f"Total well-timepoints: {total}")
print(f"  With complete ZProj: {has_zproj}")
print(f"  Missing ZProj:       {missing}")
print()
print("Missing by timepoint:")
print(inv[~inv['zproj_exists']].groupby('timepoint')['well'].count())
print()
print("Missing wells in 24h plate:")
missing_24h = inv[(inv['timepoint']=='24h') & (~inv['zproj_exists'])]['well'].tolist()
print(sorted(missing_24h)[:20], f"... total {len(missing_24h)}")
print()
print("Conditions in missing wells:", 
      inv[(inv['timepoint']=='24h') & (~inv['zproj_exists'])]['condition'].unique())


# ---- max-intensity projections for wells whose Cytation ZProj is missing (built on demand)
RAWPAT = re.compile(r'^([A-H]\d{1,2})_(\d+)_([123])_1Z(\d+)_(GFP|Oxidized roGFP2|Bright Field)_\d+\.tif$')
_raw_cache = {}
def _raw_index(tp):
    if tp not in _raw_cache:
        idx = {}
        for fname in os.listdir(plate_dirs[tp]):
            m = RAWPAT.match(fname)
            if m:
                well, field, ch_idx, z_idx, ch_name = m.groups()
                idx.setdefault(well, {}).setdefault(ch_name, []).append((int(z_idx), os.path.join(plate_dirs[tp], fname)))
        for w in idx:
            for ch in idx[w]: idx[w][ch].sort(key=lambda x: x[0])
        _raw_cache[tp] = idx
    return _raw_cache[tp]

def load_images(row, tp):
    if not row['zproj_exists']:
        ri = _raw_index(tp)[row['well']]
        return tuple(np.stack([tifffile.imread(fp) for _, fp in ri[ch]], 0).max(0)
                     for ch in ['GFP', 'Oxidized roGFP2', 'Bright Field'])
    return tifffile.imread(row['gfp_file']), tifffile.imread(row['rogfp_file']), tifffile.imread(row['bf_file'])

if os.path.exists('per_well_measurements.csv') and not os.environ.get('RECOMPUTE'):
    print('== reusing per_well_measurements.csv (set RECOMPUTE=1 to re-measure)', flush=True)
else:
    print('== segmentation + measurement (all wells)', flush=True)

    # per-well segmentation + measurement in lib/whole_spheroid.py, run in parallel worker processes
    # (SPHEROID_WORKERS, default CPU count - 1, at most 8); results are collected in inventory order.
    from parallel import WellPool
    run_lib('whole_spheroid.py')
    results = []
    qc_fails = []
    tasks = []
    for _, row in inv.iterrows():
        tp, well = row['timepoint'], row['well']
        if row['zproj_exists']:
            tasks.append((tp, well, row['condition'], (row['gfp_file'], row['rogfp_file'], row['bf_file']), None))
        else:
            tasks.append((tp, well, row['condition'], None, _raw_index(tp).get(well)))
    with WellPool(HERE, ['whole_spheroid.py'], fallback_ns=globals()) as pool:
        for status, val in pool.imap('measure_one', tasks):
            if status != 'ok':
                qc_fails.append(val); continue
            results.append(val)
            if (len(results)) % 50 == 0:
                print(f"  Processed {len(results)}/288 wells...", flush=True)

    meas = pd.DataFrame(results)
    meas.to_csv('per_well_measurements.csv', index=False)

    print(f"\nDone — {len(meas)} wells measured")
    print(f"QC fail summary:")
    print(meas.groupby(['timepoint','qc_fail'])['well'].count().to_string())
    print(f"\nLoad errors: {qc_fails}")

    meas['ratio_raw'] = meas['gfp_mean_raw'] / meas['rogfp_mean_raw']   # 488/405 inside the mask, no background subtraction
    meas.to_csv('per_well_measurements.csv', index=False)

print('== summary figures', flush=True)
exec(open(LIB('make_figs.py'), encoding='utf-8').read(), globals())
print('== representative images', flush=True)
import config as _cfg
for f in ['fig_representative_images.py', 'fig_representative_ratio_images.py']:
    exec(open(LIB(f), encoding='utf-8').read(), dict(globals(), zproj_dir=_cfg.zproj_dir))
print(f'step 1 done in {time.time()-T0:.0f}s')
