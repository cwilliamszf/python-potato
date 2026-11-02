# Pass 1: per-well 2x2-binned segmentation + background (medium) 488/405 map for flat-field estimation.
import os, re, numpy as np, pandas as pd, tifffile, scipy.ndimage as ndi, warnings
import fastops as fo   # exact cropped equivalents of the slow morphology calls
from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
warnings.filterwarnings('ignore')

BASE = DATA_DIR
PLATES = dict(PLATE_DIR)
layout = {'A':['DMSO','Fluorobrite','1uM TMZ','0.5uM OGM'],'B':['Oz L','Oz M','Oz H','1uM OGM'],
          'C':['Baf L','Baf M','Baf H','5uM OGM'],'D':['Rot L','Rot M','Rot H','20uM OGM'],
          'E':['3BP L','3BP M','3BP H','0.5uM Ogerin'],'F':['Phlor L','Phlor M','Phlor H','5uM Ogerin'],
          'G':['Olig L','Olig M','Olig H','20uM Ogerin'],'H':['Ace L','Ace M','Ace H','50uM Ogerin']}
COND = ({f'{r}{c}': LAYOUT_WELLS[r][c - 1] for r in LAYOUT_WELLS for c in range(1, 13)} if LAYOUT_WELLS
        else {f'{r}{c}': layout[r][(c-1)//3] for r in layout for c in range(1, 13)})
WELLS = [f'{r}{c}' for r in 'ABCDEFGH' for c in range(1, 13)]
RAWPAT = re.compile(r'^([A-H]\d{1,2})_\d+_([123])_1Z(\d+)_(GFP|Oxidized roGFP2|' + re.escape(BF_CHANNEL) + r')_\d+\.tif$')
CHANNELS = [('GFP', 'g'), ('Oxidized roGFP2', 'r'), (BF_CHANNEL, 'b')]

def zproj_index(plate):
    zds = [os.path.join(plate, d) for d in os.listdir(plate) if 'ZProj' in d and os.path.isdir(os.path.join(plate, d))]
    if not zds: return {}          # no Cytation projection exported: load_well builds the max projection from raw slices
    zd = zds[0]
    idx = {}
    for f in os.listdir(zd):
        w = f.split('_')[0]; p = os.path.join(zd, f)
        if 'GFP 469' in f: idx.setdefault(w, {})['g'] = p
        elif 'Oxidized roGFP2' in f: idx.setdefault(w, {})['r'] = p
        elif BF_CHANNEL in f: idx.setdefault(w, {})['b'] = p
    return idx

def raw_index(plate):
    idx = {}
    for f in os.listdir(plate):
        m = RAWPAT.match(f)
        if m:
            w, _, z, ch = m.groups()
            idx.setdefault(w, {}).setdefault(ch, []).append((int(z), os.path.join(plate, f)))
    return idx

def bin2(a):
    a = a.astype(np.float32); return a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).mean((1, 3))

def load_well(tp, w, zi, ri):
    e = zi.get(w, {})
    if all(k in e for k in 'grb'):
        g, r, b = (tifffile.imread(e[k]) for k in 'grb')
    else:
        chans = {}
        for ch, k in CHANNELS:
            sl = sorted(ri[w][ch]); st = None
            for _, fp in sl:
                im = tifffile.imread(fp); st = im if st is None else np.maximum(st, im)
            chans[k] = st
        g, r, b = chans['g'], chans['r'], chans['b']
    return bin2(g), bin2(r), bin2(b)

def seg(bf):
    inv = filters.gaussian(bf.max() - bf, sigma=2.5)
    m = inv > filters.threshold_otsu(inv)
    m = morphology.remove_small_objects(m, 250)
    for cand in (clear_border(m, buffer_size=3), m):
        L = measure.label(cand)
        if L.max():
            reg = max(measure.regionprops(L), key=lambda x: x.area)
            mk = fo.close_disk(fo.fill_holes(L == reg.label), 5)    # == binary_closing(binary_fill_holes(.), disk(5))
            return mk if 1250 <= mk.sum() <= 500_000 else None
    return None

def pass1_well(tp, w, zi_w, ri_w):
    """One well of the legacy first spatial pass (worker entry point, lib/parallel.py).
    Returns (record, mask or None, background-ratio map or None)."""
    rec = dict(timepoint=tp, well=w, condition=COND[w], status='ok')
    try:
        g, r, b = load_well(tp, w, {w: zi_w}, {w: ri_w} if ri_w is not None else {})
    except PermissionError:
        rec['status'] = 'locked'; return rec, None, None
    m = seg(b)
    ratio = g / np.maximum(r, 1)
    bgmask = ~fo.dilate_iter(m, 40) if m is not None else np.ones_like(g, bool)   # == ~ndi.binary_dilation(m, iterations=40)
    rb = np.where(bgmask, ratio, np.nan)
    rb8 = fo.nanmedian_small(rb.reshape(245, 4, 245, 4).transpose(0, 2, 1, 3).reshape(245, 245, 16), 2)   # == np.nanmedian(., axis=2)
    bg = (rb8 / np.nanmedian(rb8)).astype(np.float32)
    rec['bg_ratio_median'] = float(np.nanmedian(rb))
    if m is None:
        rec['status'] = 'seg_fail'
    return rec, m, bg

def run_pass1():
    from parallel import WellPool
    masks, bgmaps, rows = {}, {}, []
    tasks = []
    for tp, plate in PLATES.items():
        zi = zproj_index(plate); ri = None
        for w in WELLS:
            if not all(k in zi.get(w, {}) for k in 'grb') and ri is None:
                ri = raw_index(plate)
            tasks.append((tp, w, zi.get(w, {}), ri.get(w) if ri is not None else None))
    with WellPool(HERE, ['common.py'], fallback_ns=globals()) as pool:
        for (tp, w, _, _), (rec, m, bg) in zip(tasks, pool.imap('pass1_well', tasks)):
            if bg is not None: bgmaps[(tp, w)] = bg
            if m is not None: masks[(tp, w)] = m
            rows.append(rec)
            if w == WELLS[-1]: print(tp, 'done', flush=True)
    return masks, bgmaps, pd.DataFrame(rows)
