import fastops as fo   # exact cropped equivalents of the slow full-resolution morphology
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; matplotlib.rcParams['svg.fonttype'] = 'none'

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
plate_dirs = {
    '24h': PLATE_DIR['24h'],
    '48h': PLATE_DIR['48h'],
    '72h': PLATE_DIR['72h'],
}

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


import tifffile, numpy as np
zproj_cache_24h = {}
for w in ['E7']:
    zproj_cache_24h[w] = {key: np.stack([tifffile.imread(fp) for _, fp in sorted(RI24[w][ch])]).max(0)
                          for ch, key in [('GFP','gfp'), ('Oxidized roGFP2','rogfp'), ('Bright Field','bf')]}


from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
import scipy.ndimage as ndi

def load_images(row, tp='48h'):
    """Load gfp, rogfp, bf arrays for a given inventory row."""
    if tp == '24h' and row['well'] in zproj_cache_24h:
        d = zproj_cache_24h[row['well']]
        return d['gfp'], d['rogfp'], d['bf']
    return (tifffile.imread(row['gfp_file']),
            tifffile.imread(row['rogfp_file']),
            tifffile.imread(row['bf_file']))

def segment_spheroid(bf_img, area_min=5_000, area_max=1_500_000):
    """
    Segment spheroid from bright-field ZProj.
    Returns (mask, props_dict) or (None, {'qc_fail': reason})
    """
    # Invert so spheroid is bright
    inv = bf_img.max() - bf_img.astype(np.float32)
    # Gaussian blur
    blurred = filters.gaussian(inv, sigma=5)
    # Otsu threshold
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    # Remove small objects and objects touching border
    binary = morphology.remove_small_objects(binary, min_size=1000)
    binary = clear_border(binary, buffer_size=5)
    # Label and pick largest
    labeled = measure.label(binary)
    if labeled.max() == 0:
        return None, {'qc_fail': 'no_objects'}
    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    # Fill holes
    mask = fo.fill_holes(labeled == largest.label)          # == ndi.binary_fill_holes
    # Morphological closing
    mask = fo.close_disk(mask, 10)                          # == binary_closing(mask, disk(10))
    # Area QC
    area = mask.sum()
    if area < area_min:
        return None, {'qc_fail': f'too_small_{area}'}
    if area > area_max:
        return None, {'qc_fail': f'too_large_{area}'}
    # Compute props
    props = measure.regionprops(mask.astype(int))[0]
    equiv_diam = props.equivalent_diameter_area
    perimeter  = props.perimeter
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    return mask, {'area_px': area, 'equiv_diam_px': equiv_diam,
                  'circularity': circularity, 'qc_fail': None}

# Test on DMSO A1 at 48h
row_dmso = inv[(inv['timepoint']=='48h') & (inv['well']=='A1')].iloc[0]
gfp_t, rogfp_t, bf_t = load_images(row_dmso, '48h')
mask, props = segment_spheroid(bf_t)

print(f"DMSO A1 48h: area={props['area_px']:,} px², diam={props['equiv_diam_px']:.1f} px, circularity={props['circularity']:.3f}")

# Also test a Baf H (C7) well
row_baf = inv[(inv['timepoint']=='48h') & (inv['well']=='C7')].iloc[0]
gfp_b, rogfp_b, bf_b = load_images(row_baf, '48h')
mask_b, props_b = segment_spheroid(bf_b)
print(f"Baf H  C7 48h: area={props_b['area_px']:,} px², diam={props_b['equiv_diam_px']:.1f} px, circularity={props_b['circularity']:.3f}")


from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
import scipy.ndimage as ndi

def segment_spheroid_robust(bf_img, area_min=5_000, area_max=2_000_000):
    """Segment spheroid; fall back to no-border-clear if Otsu yields nothing."""
    inv = (bf_img.max() - bf_img.astype(np.float32))
    blurred = filters.gaussian(inv, sigma=5)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    binary = morphology.remove_small_objects(binary, min_size=1000)

    def _pick_largest(b):
        labeled = measure.label(b)
        if labeled.max() == 0:
            return None
        regions = measure.regionprops(labeled)
        largest = max(regions, key=lambda r: r.area)
        mask = fo.fill_holes(labeled == largest.label)          # == ndi.binary_fill_holes
        mask = fo.close_disk(mask, 10)                          # == binary_closing(mask, disk(10))
        return mask

    # Try with border clearing first
    mask = _pick_largest(clear_border(binary, buffer_size=5))
    # Fall back: no border clear
    if mask is None:
        mask = _pick_largest(binary)

    if mask is None:
        return None, {'qc_fail': 'no_objects', 'area_px': 0, 'equiv_diam_px': 0, 'circularity': 0}

    area = int(mask.sum())
    if area < area_min:
        return None, {'qc_fail': f'too_small', 'area_px': area, 'equiv_diam_px': 0, 'circularity': 0}
    if area > area_max:
        return None, {'qc_fail': f'too_large', 'area_px': area, 'equiv_diam_px': 0, 'circularity': 0}

    props = measure.regionprops(mask.astype(np.uint8))[0]
    equiv_diam  = props.equivalent_diameter_area
    perimeter   = props.perimeter
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    return mask, {'qc_fail': None, 'area_px': area,
                  'equiv_diam_px': float(equiv_diam), 'circularity': float(circularity)}



from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# Representative wells: 2 per row: DMSO, TMZ, Baf H, Olig H, Rot H, OGM
qc_wells = [
    ('48h', 'A1',  'DMSO'),
    ('48h', 'A7',  '1uM TMZ'),
    ('48h', 'C7',  'Baf H'),
    ('48h', 'C9',  'Baf H rep2'),
    ('48h', 'D7',  'Rot H'),
    ('48h', 'G7',  'Olig H'),
    ('48h', 'D10', '20uM OGM'),
    ('48h', 'H10', '50uM Ogerin'),
    ('72h', 'A1',  'DMSO 72h'),
    ('72h', 'C7',  'Baf H 72h'),
    ('72h', 'G7',  'Olig H 72h'),
    ('24h', 'E7',  '3BP H 24h'),
]

fig, axes = plt.subplots(len(qc_wells), 4, figsize=(16, len(qc_wells)*3.2))
fig.suptitle('Segmentation QC — bright field | GFP | mask overlay | ratio', fontsize=12, y=1.01)

def pct_stretch(arr, lo=1, hi=99):
    vmin, vmax = np.percentile(arr, [lo, hi])
    return np.clip((arr.astype(float) - vmin) / (vmax - vmin + 1e-9), 0, 1)

for i, (tp, well, label) in enumerate(qc_wells):
    row_df = inv[(inv['timepoint']==tp) & (inv['well']==well)]
    if row_df.empty:
        for j in range(4): axes[i, j].axis('off')
        continue
    row_df = row_df.iloc[0]
    gfp_i, rogfp_i, bf_i = load_images(row_df, tp)
    mask_i, props_i = segment_spheroid(bf_i)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_i = np.where(rogfp_i > 0, gfp_i.astype(float)/rogfp_i.astype(float), np.nan)
    
    axes[i,0].imshow(pct_stretch(bf_i), cmap='gray'); axes[i,0].axis('off')
    axes[i,0].set_title(f'{label} ({well})', fontsize=8, loc='left')
    
    axes[i,1].imshow(pct_stretch(gfp_i), cmap='Greens'); axes[i,1].axis('off')
    
    # Mask overlay on BF
    axes[i,2].imshow(pct_stretch(bf_i), cmap='gray')
    if mask_i is not None:
        contours = measure.find_contours(mask_i, 0.5)
        for c in contours:
            axes[i,2].plot(c[:,1], c[:,0], 'r-', lw=1.5)
        status = f"A={props_i['area_px']//1000}k px²  circ={props_i['circularity']:.2f}"
    else:
        status = f"FAIL: {props_i.get('qc_fail')}"
    axes[i,2].set_title(status, fontsize=7, loc='left', color='red' if mask_i is None else 'k')
    axes[i,2].axis('off')
    
    # Ratio image (masked)
    ratio_masked = np.where(mask_i if mask_i is not None else np.zeros_like(ratio_i, bool),
                             ratio_i, np.nan)
    vmed = np.nanmedian(ratio_masked)
    axes[i,3].imshow(ratio_masked, cmap='RdBu_r',
                     vmin=vmed*0.85, vmax=vmed*1.15)
    axes[i,3].axis('off')

plt.tight_layout()
fig.savefig('segmentation_qc.svg', dpi=110, bbox_inches='tight')
plt.close(fig)
print("segmentation_qc.png saved")


import tifffile
import numpy as np
import scipy.ndimage as ndi
from skimage import filters

BASE = DATA_DIR
plate_48h_root = PLATE_DIR['48h']
import config as _cfg
plate_48h_zproj = _cfg.zproj_dir('48h')

import re
raw_pattern_48 = re.compile(
    r'^([A-H]\d{1,2})_(\d+)_([123])_1Z(\d+)_(GFP|Oxidized roGFP2|Bright Field)_\d+\.tif$'
)
raw_index_48 = {}
for fname in os.listdir(plate_48h_root):
    m = raw_pattern_48.match(fname)
    if not m:
        continue
    well, field, ch_idx, z_idx, ch_name = m.groups()
    z_idx = int(z_idx)
    raw_index_48.setdefault(well, {}).setdefault(ch_name, []).append(
        (z_idx, os.path.join(plate_48h_root, fname)))
for well in raw_index_48:
    for ch in raw_index_48[well]:
        raw_index_48[well][ch].sort(key=lambda x: x[0])

def focus_score_laplacian(img):
    """Variance of Laplacian — higher = better focus."""
    lap = ndi.laplace(img.astype(float))
    return float(lap.var())

def load_stack(well, channel, index=raw_index_48):
    """Load all z-slices for a channel; return (n_z, H, W) array."""
    slices = index[well][channel]
    return np.stack([tifffile.imread(fp) for _, fp in slices], axis=0)

def compute_projections(well):
    """
    Returns dict with:
      methodA  — max(GFP) / max(roGFP2) per pixel  (current approach = what ZProj produces)
      methodB  — max(GFP_z / roGFP2_z) per pixel   (ratio-first, max-project the ratio)
      methodC  — GFP_z* / roGFP2_z*  at best-focus BF z-plane (focal-plane select)
      best_z   — index of best-focus BF slice
    """
    gfp_stack   = load_stack(well, 'GFP')
    rogfp_stack = load_stack(well, 'Oxidized roGFP2')
    bf_stack    = load_stack(well, 'Bright Field')

    n_z = bf_stack.shape[0]

    # Best-focus BF plane
    scores = np.array([focus_score_laplacian(bf_stack[z]) for z in range(n_z)])
    best_z = int(np.argmax(scores))

    # Method A: independent max projection then ratio
    gfp_maxproj   = gfp_stack.max(axis=0).astype(float)
    rogfp_maxproj = rogfp_stack.max(axis=0).astype(float)
    methodA = np.where(rogfp_maxproj > 0, gfp_maxproj / rogfp_maxproj, np.nan)

    # Method B: ratio at each z, then max-project the ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_per_z = np.where(rogfp_stack > 0,
                                gfp_stack.astype(float) / rogfp_stack.astype(float), np.nan)
    methodB = np.nanmax(ratio_per_z, axis=0)

    # Method C: ratio at best-focus BF z-plane
    gfp_z   = gfp_stack[best_z].astype(float)
    rogfp_z = rogfp_stack[best_z].astype(float)
    methodC = np.where(rogfp_z > 0, gfp_z / rogfp_z, np.nan)

    return dict(methodA=methodA, methodB=methodB, methodC=methodC,
                best_z=best_z, bf_focus_scores=scores)

# Pilot wells: 2 DMSO, 2 Baf H, 2 OGM 20µM
pilot_wells = {
    'A1': 'DMSO',  'A3': 'DMSO',
    'C7': 'Baf H', 'C9': 'Baf H',
    'D10':'20µM OGM', 'D12':'20µM OGM',
}
print("Loading z-stacks and computing projections (pilot)...")
proj_results = {}
for well in pilot_wells:
    proj_results[well] = compute_projections(well)
    print(f"  {well} ({pilot_wells[well]}): best_z={proj_results[well]['best_z']}, "
          f"focus_range=[{proj_results[well]['bf_focus_scores'].min():.0f}–{proj_results[well]['bf_focus_scores'].max():.0f}]")
print("Done.")


apply_figure_style(sizes=(7, 6, 5.5))

# ── Compute masked mean ratio for each method ─────────────────────────────
# We need BF ZProj for segmentation (use existing ZProj)
compare_rows = []
for well, cond in pilot_wells.items():
    row_df = inv[(inv['timepoint']=='48h') & (inv['well']==well)]
    if row_df.empty: continue
    row_df = row_df.iloc[0]

    # Load BF from ZProj for segmentation
    _, _, bf_zproj = load_images(row_df, '48h')
    mask, props = segment_spheroid_robust(bf_zproj)
    if mask is None:
        print(f"  {well}: segmentation failed — {props['qc_fail']}")
        continue

    res = proj_results[well]

    # Also get the ratio at z=0 and z=best_z for method C
    bz = res['best_z']
    gfp_stack   = np.stack([tifffile.imread(fp) for _, fp in raw_index_48[well]['GFP']],          axis=0)
    rogfp_stack = np.stack([tifffile.imread(fp) for _, fp in raw_index_48[well]['Oxidized roGFP2']], axis=0)
    
    # Method C: best BF z-plane
    methodC_img = res['methodC']    # already computed

    for method_key, img in [('A_maxproj_ratio',  res['methodA']),
                              ('B_ratio_maxproj',  res['methodB']),
                              ('C_focal_z',        methodC_img)]:
        val = float(np.nanmean(img[mask]))
        compare_rows.append({'well': well, 'condition': cond, 'method': method_key,
                              'mean_ratio': val, 'best_z': bz})

cdf = pd.DataFrame(compare_rows)
pivot = cdf.pivot_table(index=['well','condition'], columns='method', values='mean_ratio')
pivot.columns.name = None
pivot['B_vs_A_%'] = (pivot['B_ratio_maxproj'] / pivot['A_maxproj_ratio'] - 1) * 100
pivot['C_vs_A_%'] = (pivot['C_focal_z']       / pivot['A_maxproj_ratio'] - 1) * 100
print(pivot.round(4).to_string())

# ── Figure: side-by-side comparison ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: absolute mean ratios
x = np.arange(len(pilot_wells))
width = 0.26
methods = ['A_maxproj_ratio', 'B_ratio_maxproj', 'C_focal_z']
method_labels = ['A: max(GFP)/max(roGFP2)\n(current, = Cytation ZProj)', 
                  'B: max(GFP/roGFP2)\n(ratio-first, then max-proj)', 
                  'C: GFP/roGFP2 at best-BF z\n(focal-plane select)']
colors = ['#1F77B4', '#FF7F0E', '#2CA02C']

for mi, (mkey, mlabel, color) in enumerate(zip(methods, method_labels, colors)):
    vals = [pivot.loc[(w, c), mkey] if (w, c) in pivot.index else np.nan
            for w, c in pilot_wells.items()]
    axes[0].bar(x + (mi - 1) * width, vals, width, label=mlabel, color=color, alpha=0.85)

axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{w}\n({c})" for w, c in pilot_wells.items()], fontsize=6)
axes[0].set_ylabel('Mean ratio (GFP/roGFP2) within spheroid mask')
axes[0].set_title('Absolute ratio by method (48 h pilot)', fontsize=8)
axes[0].legend(fontsize=5.5, loc='upper right')

# Right: % deviation from Method A
for mi, (mkey, mlabel, color) in enumerate([('B_vs_A_%','B vs A','#FF7F0E'),
                                              ('C_vs_A_%','C vs A','#2CA02C')]):
    vals = [pivot.loc[(w, c), mkey] if (w, c) in pivot.index else np.nan
            for w, c in pilot_wells.items()]
    axes[1].bar(x + (mi - 0.5) * 0.35, vals, 0.32, label=mlabel, color=color, alpha=0.85)

axes[1].axhline(0, color='0.4', lw=0.8, ls='--')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{w}\n({c})" for w, c in pilot_wells.items()], fontsize=6)
axes[1].set_ylabel('% difference from Method A')
axes[1].set_title('Deviation from current method (Method A)', fontsize=8)
axes[1].legend(fontsize=6)

plt.suptitle('Pilot comparison: max-proj ratio vs ratio-first methods (48 h, 6 wells)', fontsize=8.5, y=1.02)
plt.tight_layout()
fig.savefig('projection_method_comparison.svg', dpi=130, bbox_inches='tight')
plt.close(fig)
print("\nprojection_method_comparison.png saved")
