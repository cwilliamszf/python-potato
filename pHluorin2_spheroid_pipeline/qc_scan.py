"""QC scan over all spheroid images of the three experiments.
(1) detector saturation (Cytation SaturationLevel = 65504) in 488, 405 and bright field
(2) fluorescence medium background vs camera floor and spheroid signal
(3) segmentation: v1 (bright-field max, Otsu) and v2 (bright-field min, centre object) vs a fluorescence reference mask
Usage: python qc_scan.py <out.csv> [shard/n_shards]
Images are scanned in parallel worker processes (SPHEROID_WORKERS, default CPU count - 1, at most 8); rows are
written in task order. SPHEROID_QC_BASES=<file.json> replaces BASES below ({exp: [data_dir, {timepoint: folder}]}),
and SPHEROID_QC_EXPS=exp1,exp2 restricts the scan to some experiments."""
import os, re, sys, json, numpy as np, pandas as pd, tifffile, scipy.ndimage as ndi
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import fastops as fo          # exact cropped equivalents of the slow morphology calls
from parallel import n_workers
mo = morphology
SAT = 65504
BASES = {'MCF7_exp1': (r'C:\path\to\data\261001_105300_10.1.26 MCF7 P6 Spheroid Drug Assay',
                       {'24h': '261001_105300_Plate 1', '48h': '261002_173508_Plate 3', '72h': '261003_211651_Plate 1'}),
         'MCF7_exp2': (r'C:\path\to\data\261012_193330_MCF7pHluorinpilotdrugsrepeat',
                       {'2h': '261012_193330_Plate 1', '24h': '261013_084820_Plate 2', '48h': '261014_105148_Plate 3', '72h': '261015_095705_Plate 4'}),
         'HEK293': (r'C:\path\to\data\261012_184423_293phluorinpilotdrugs',
                    {'2h': '261012_190058_Plate 1', '24h': '261013_092344_Plate 2', '48h': '261014_095240_Plate 3', '72h': '261015_083507_Plate 4'}),
         'T47D_exp5': (r'C:\path\to\data\261019_155302_T47DpHluorindrugscreenpilot',
                       {'24h': '261019_155302_Plate 1', '48h': '261020_092906_Plate 2', '72h': '261021_092749_Plate 3'})}
if os.environ.get('SPHEROID_QC_BASES'):
    BASES = {k: (v[0], dict(v[1])) for k, v in json.load(open(os.environ['SPHEROID_QC_BASES'], encoding='utf-8')).items()}
# transmitted-light channel per experiment (the T47D plates were imaged in phase contrast)
BF = {'T47D_exp5': 'Phase Contrast'}
def rawpat(exp):
    return re.compile(r'^([A-H]\d{1,2})_(\d+)_[123]_1Z(\d+)_(GFP|Oxidized roGFP2|'
                      + re.escape(BF.get(exp, 'Bright Field')) + r')_\d+\.tif$')
def chmap(exp):
    return {'GFP': 'g', 'Oxidized roGFP2': 'r', BF.get(exp, 'Bright Field'): 'b'}
CH = chmap('')

def bin2(a):
    a = a.astype(np.float32); return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))

def seg_v1(bf):
    inv = filters.gaussian(bf.max() - bf, sigma=2.5)
    m = inv > filters.threshold_otsu(inv)
    m = mo.remove_small_objects(m, 250)
    for cand in (clear_border(m, buffer_size=3), m):
        L = measure.label(cand)
        if L.max():
            reg = max(measure.regionprops(L), key=lambda x: x.area)
            mk = fo.close_disk(fo.fill_holes(L == reg.label), 5)    # == binary_closing(binary_fill_holes(.), disk(5))
            return mk if 1250 <= mk.sum() <= 500_000 else None
    return None

def seg_v2(b):
    flat = b / ndi.gaussian_filter(b, 60)
    sm = ndi.gaussian_filter(flat, 2)
    core = sm[(sm > np.percentile(sm, 0.5)) & (sm < np.percentile(sm, 99.5))]
    m = sm < min(filters.threshold_otsu(core), 0.92)
    m = mo.binary_opening(m, mo.disk(2)); m = ndi.binary_fill_holes(mo.binary_closing(m, mo.disk(4)))
    L = measure.label(m)
    regs = [x for x in measure.regionprops(L) if 600 <= x.area <= 400_000]
    if not regs: return None
    cy, cx = np.array(m.shape) / 2
    reg = min(regs, key=lambda x: np.hypot(x.centroid[0] - cy, x.centroid[1] - cx) / (np.sqrt(x.area) + 1) - 0.5 * x.solidity)
    return fo.fill_holes(fo.close_disk(L == reg.label, 6))   # == binary_fill_holes(binary_closing(., disk(6)))

def seg_fluo(g, r):
    """Reference mask from fluorescence: log(488+405) smoothed, Otsu, object nearest centre, filled."""
    s = ndi.gaussian_filter(np.log(np.maximum(g + r, 1)), 3)
    m = s > filters.threshold_otsu(s)
    m = mo.binary_opening(m, mo.disk(3)); L = measure.label(m)
    regs = [x for x in measure.regionprops(L) if x.area >= 600]
    if not regs: return None
    cy, cx = np.array(m.shape) / 2
    reg = min(regs, key=lambda x: np.hypot(x.centroid[0] - cy, x.centroid[1] - cx) / (np.sqrt(x.area) + 1))
    return fo.fill_holes(fo.close_disk(L == reg.label, 4))   # == binary_fill_holes(binary_closing(., disk(4)))

def dice(a, b):
    if a is None or b is None: return np.nan
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))

def task(args):
    exp, tp, w, files, zfiles, need_min = args
    CH = chmap(exp); BFCH = BF.get(exp, 'Bright Field')
    out = dict(exp=exp, timepoint=tp, well=w)
    mx, mn, nslices = {}, {}, {}
    if zfiles is not None:                     # Cytation ZProj (pixel-exact max projection)
        for k, fp in zfiles.items(): mx[k] = tifffile.imread(fp)
    for ch, sl in files.items():
        k = CH[ch]
        if zfiles is not None and k in mx and not (k == 'b' and need_min): continue
        a_max = a_min = None; nsat_slices = 0
        for _, fp in sorted(sl):
            im = tifffile.imread(fp)
            nsat_slices += int((im >= SAT).sum())
            a_max = im if a_max is None else np.maximum(a_max, im); a_min = im if a_min is None else np.minimum(a_min, im)
        mx[k] = a_max if k not in mx else mx[k]; mn[k] = a_min; nslices[k] = len(sl)
        out[f'{k}_sat_px_allslices'] = nsat_slices
    for k in 'grb':
        a = mx[k]
        out[f'{k}_max'] = int(a.max()); out[f'{k}_p9999'] = float(np.percentile(a, 99.99))
        out[f'{k}_sat_px'] = int((a >= SAT).sum()); out[f'{k}_near_sat_px'] = int((a >= 0.95 * SAT).sum())
        out[f'{k}_px_at_max'] = int((a >= a.max() - max(1, 0.002 * a.max())).sum())   # pile-up at the top = clipping
    g2, r2 = bin2(mx['g']), bin2(mx['r'])
    m1 = seg_v1(bin2(mx['b']))
    m2 = seg_v2(bin2(mn['b'])) if 'b' in mn else None
    mf = seg_fluo(g2, r2)
    for nm, m in [('v1', m1), ('v2', m2), ('fl', mf)]:
        out[f'area_{nm}'] = int(m.sum()) if m is not None else np.nan
    out['dice_v1_fl'] = dice(m1, mf); out['dice_v2_fl'] = dice(m2, mf); out['dice_v1_v2'] = dice(m1, m2)
    ref = mf if mf is not None else (m2 if m2 is not None else m1)
    if ref is not None:
        far = ~fo.dilate_iter(ref, 25)          # == ~ndi.binary_dilation(ref, iterations=25)
        out['g_medium'] = float(np.median(g2[far])); out['r_medium'] = float(np.median(r2[far]))
        out['g_sph'] = float(np.mean(g2[ref])); out['r_sph'] = float(np.mean(r2[ref]))
        out['g_medium_sd'] = float(np.std(g2[far])); out['r_medium_sd'] = float(np.std(r2[far]))
        satg = bin2((mx['g'] >= 0.95 * SAT).astype(np.float32)) > 0; satr = bin2((mx['r'] >= 0.95 * SAT).astype(np.float32)) > 0
        out['g_nearsat_in_sph'] = int((satg & ref).sum()); out['r_nearsat_in_sph'] = int((satr & ref).sum())
        out['b_medium'] = float(np.median(bin2(mx['b'])[far]))
        if 'b' in mn: out['b_min_medium'] = float(np.median(bin2(mn['b'])[far]))
    pk = {nm: (np.packbits(m) if m is not None else None) for nm, m in [('v1', m1), ('v2', m2), ('fl', mf)]}
    return out, pk

def task_safe(t):
    try:
        return task(t)
    except Exception as e:
        return dict(exp=t[0], timepoint=t[1], well=t[2], error=repr(e)[:200]), None

def build_tasks():
    T = []
    only = [e for e in os.environ.get('SPHEROID_QC_EXPS', '').split(',') if e]
    for exp, (base, pf) in BASES.items():
        if only and exp not in only: continue
        PAT = rawpat(exp); BFCH = BF.get(exp, 'Bright Field')
        for tp, fo in pf.items():
            p = os.path.join(base, fo); idx = {}
            for f in os.listdir(p):
                mm = PAT.match(f)
                if mm: idx.setdefault(mm.group(1), {}).setdefault(mm.group(4), []).append((int(mm.group(3)), os.path.join(p, f)))
            zd = [os.path.join(p, d) for d in os.listdir(p) if 'ZProj' in d and os.path.isdir(os.path.join(p, d))]
            zi = {}
            if zd:
                for f in os.listdir(zd[0]):
                    w = f.split('_')[0]; k = 'g' if 'GFP 469' in f else 'r' if 'Oxidized' in f else 'b' if BFCH in f else None
                    if k: zi.setdefault(w, {})[k] = os.path.join(zd[0], f)
            for w in sorted(idx):
                z = zi.get(w); z = z if z is not None and len(z) == 3 else None
                need_min = exp != 'MCF7_exp1' or tp == '48h'        # exp1: bright-field min projection only on the 48 h plate
                files = idx[w] if (z is None or need_min) else {}
                if z is not None and need_min: files = {BFCH: idx[w][BFCH]}
                T.append((exp, tp, w, files, z, need_min))
    return T

if __name__ == '__main__':
    out_csv = sys.argv[1]; shard, nsh = (int(x) for x in sys.argv[2].split('/')) if len(sys.argv) > 2 else (0, 1)
    T = build_tasks()[shard::nsh]; print(len(T), 'tasks', flush=True)
    rows, masks = [], {}
    nw = n_workers()
    ex = ProcessPoolExecutor(nw) if nw > 1 else None
    results = ex.map(task_safe, T, chunksize=2) if ex else map(task_safe, T)
    for i, (t, (r, pk)) in enumerate(zip(T, results)):
        k = t[:3]
        rows.append(r)
        for nm, v in (pk or {}).items():
            if v is not None: masks[f'{k[0]}|{k[1]}|{k[2]}|{nm}'] = v
        if (i + 1) % 20 == 0: print(i + 1, flush=True)
    if ex: ex.shutdown()
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    np.savez_compressed(out_csv.replace('.csv', '_masks.npz'), **masks)
    print('done', flush=True)
