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

"""Step 4 - background models and spatial features of the 488/405 maps (HIGHER = MORE ACIDIC).
  * camera/detector floor per plate from raw z-slices
  * flat-fields (ratio, per-channel medium, excitation ratio) from spheroid-free medium
  * calibrated background fraction alpha per plate (pixel ratio uncorrelated with brightness in null conditions)
  * two-pass feature extraction with removal of the image-fixed shared residual pattern
  * robust heterogeneity, low-signal flag, QC exclusions
Outputs: calibration.json, background_alpha_scan.csv, spatial_features_v2.csv, spatial_maps_cal.pkl, radial_profiles.pkl"""

run_lib('common.py'); run_lib('stage1_extract.py'); run_lib('stage2_features.py')
with np.load('stage1_arrays.npz') as Z:
    keys = [tuple(k.split('|')) for k in Z['keys']]; gA = Z['g']; rA = Z['r']; mA = Z['m']
s1 = pd.read_csv('stage1_status.csv')
okset = set(zip(s1[s1.status == 'ok'].timepoint, s1[s1.status == 'ok'].well))
G = {k: gA[i] for i, k in enumerate(keys)}; Rr = {k: rA[i] for i, k in enumerate(keys)}
M = {k: mA[i] for i, k in enumerate(keys) if k in okset}
EXCL = {(t, w) for t in TPS for w in EXCLUDE_WELLS_ALL_TP}
keys_ok = [k for k in M if k not in EXCL]

print('== detector floor', flush=True)

floors = {}
for tp in TPS:
    ri = raw_index(PLATES[tp]); vals = {'GFP': [], 'Oxidized roGFP2': []}
    for w in ['A2', 'B5', 'C11', 'E2', 'F8', 'G5', 'H2', 'D5']:
        if w not in ri: continue
        for ch in vals:
            sl = sorted(ri[w][ch])
            # minimum over a few slices of the 0.1th percentile, then binned-4x4 equivalent (median filter-ish)
            v = [np.percentile(bin2f(bin2f(tifffile.imread(fp))), 0.5) for _, fp in sl[::9]]
            vals[ch].append(min(v))
    floors[tp] = (float(np.median(vals['GFP'])), float(np.median(vals['Oxidized roGFP2'])))
print({k: tuple(round(x) for x in v) for k, v in floors.items()})

fields, med = build_fields(G, Rr, M, keys_ok, floors)
print('== background-fraction scan', flush=True)

ctrl = set(CTRL_CONDS_CFG) if CTRL_CONDS_CFG else {'DMSO', 'Fluorobrite', 'Phlor L', 'Phlor M', 'Phlor H', '0.5uM OGM', '0.5uM Ogerin'}
alphas = np.round(np.linspace(0, 1, 11), 1)
rows = []
for kk in keys_ok:
    c = COND[kk[1]]
    if c not in ctrl: continue
    F_ = fields[kk[0]]; g = G[kk].astype(np.float32); r = Rr[kk].astype(np.float32); m = M[kk]
    d = ndi.distance_transform_edt(m); V = m & (d > EDGE)
    fg0, fr0 = F_['floor']; lg, lr = med[kk]
    for a in alphas:
        Bg = fg0 + a * (lg * F_['g'] - fg0); Br = fr0 + a * (lr * F_['r'] - fr0)
        sg, sr = g - Bg, r - Br; ok = V & (sg > 0) & (sr > 0)
        if ok.sum() < 200: rows.append(dict(key=kk, cond=c, alpha=a, rho=np.nan, pos_frac=ok.sum()/V.sum())); continue
        Ra = sg[ok] / sr[ok] / F_['exratio'][ok]
        bright = np.log(sg[ok] + sr[ok])
        rows.append(dict(key=kk, cond=c, alpha=a, rho=stats.spearmanr(Ra, bright)[0], pos_frac=ok.sum()/V.sum(),
                         med_ratio=np.median(Ra)))
A = pd.DataFrame(rows)
print(A.groupby('alpha')[['rho', 'pos_frac', 'med_ratio']].median().round(3).to_string())
print("n wells:", A.key.nunique())


A['tp'] = [k[0] for k in A.key]
ALPHA = {}
for tp, d in A.groupby('tp'):
    mr = d.groupby('alpha')['rho'].median()
    span = float(mr.max() - mr.min())
    if ALPHA_FIXED is not None and (ALPHA_FORCE or span < ALPHA_MIN_SPAN or not (mr.min() < 0 < mr.max())):
        # brightness-ratio correlation does not cross zero / barely depends on alpha (bright spheroids, background a
        # small fraction of signal): alpha is not identifiable from these data -> use the configured value
        ALPHA[tp] = float(ALPHA_FIXED); print(f'{tp}: alpha not identifiable (rho span {span:.2f}); using fixed alpha = {ALPHA_FIXED}')
    else:
        ALPHA[tp] = float(np.interp(0, mr.values, mr.index.values))
# bootstrap CI over wells
rng = np.random.default_rng(1); boot = {tp: [] for tp in TPS}
for tp, d in A.groupby('tp'):
    ks = d.key.unique()
    for _ in range(300):
        sub = d[d.key.isin(rng.choice(ks, len(ks)))].groupby('alpha')['rho'].median()
        boot[tp].append(np.interp(0, sub.values, sub.index.values))
print({tp: (round(ALPHA[tp], 2), tuple(np.round(np.percentile(boot[tp], [2.5, 97.5]), 2))) for tp in TPS})
ALPHA_INFO = {tp: dict(alpha=ALPHA[tp], rho_span=float(A[A.tp == tp].groupby('alpha').rho.median().pipe(lambda s: s.max() - s.min())), fixed=bool(ALPHA_FIXED is not None and ALPHA[tp] == ALPHA_FIXED)) for tp in TPS}
A.to_csv('background_alpha_scan.csv', index=False)

json.dump({'floors': floors, 'alpha': ALPHA, 'alpha_info': ALPHA_INFO}, open('calibration.json', 'w'), indent=1)
print('== feature extraction (two passes)', flush=True)
for tp in TPS: fields[tp]['alpha'] = ALPHA[tp]
t0 = time.time()
# both passes run per spheroid in parallel worker processes (lib/parallel.py; SPHEROID_WORKERS); results are
# collected in keys_ok order, so every table is identical to a serial run
from parallel import WellPool
resA = {'': {}, '_cal': {}, '_log': {}, '_cal_log': {}}
jobs = [(kk, G[kk], Rr[kk], M[kk], med[kk], 1) for kk in keys_ok]
with WellPool(HERE, ['stage2_features.py'], shared={'PAR_FIELDS': fields}, fallback_ns=globals()) as pool:
    for kk, res in zip(keys_ok, pool.imap('well_features_job', jobs, chunksize=2)):
        for tg in ['', '_cal']:
            resA[tg][kk] = res[tg][0]; resA[tg + '_log'][kk] = res[tg][1]
print(f'   pass 1: {time.time()-t0:.0f}s', flush=True)
shared = {}
for tp in TPS:
    shared[tp] = {}
    for tg in resA:
        st = np.stack([v.astype(np.float32) for kk, v in resA[tg].items() if kk[0] == tp])
        cov = np.isfinite(st).sum(0); sh = fo.nanmedian_small(st, 0); sh[cov < 10] = 0; shared[tp][tg] = sh.astype(np.float32)   # == np.nanmedian(st, 0)
del resA, st; gc.collect()
rows, MAPS = [], {}
jobs = [(kk, G[kk], Rr[kk], M[kk], med[kk], 2) for kk in keys_ok]
with WellPool(HERE, ['stage2_features.py'], shared={'PAR_FIELDS': fields, 'PAR_SHARED': shared}, fallback_ns=globals()) as pool:
    for kk, (o, mp) in zip(keys_ok, pool.imap('well_features_job', jobs, chunksize=2)):
        o.update(timepoint=kk[0], well=kk[1], condition=COND[kk[1]]); rows.append(o); MAPS[kk] = mp
print(f'   pass 2: {time.time()-t0:.0f}s', flush=True)
F2 = pd.DataFrame(rows)
info = F2['condition'].apply(cond_info)
F2['compound'] = [i[0] for i in info]; F2['dose_um'] = [i[1] for i in info]; F2['dose_rank'] = [i[2] for i in info]
F2 = F2.merge(s1[['timepoint','well','mask_split']], on=['timepoint','well'], how='left')
PROF = {c: dict(zip(zip(F2.timepoint, F2.well), F2[c])) for c in F2 if c.startswith('_prof')}
Fv = F2.drop(columns=[c for c in F2 if c.startswith('_prof')]); Fv['alpha'] = Fv.timepoint.map(ALPHA)
Fv.to_csv('spatial_features_v2.csv', index=False)

def robust_het(mp):
    res = mp['res'].astype(np.float32); V = np.isfinite(res)
    v = res[V]; mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    clipped = np.clip(res, -5 * mad, 5 * mad)
    return float(mad), corr_length(np.where(V, clipped, 0), V)
for tg in ['', '_cal']:
    out = [robust_het(MAPS[(t, w)][tg]) for t, w in zip(Fv.timepoint, Fv.well)]
    Fv['resid_sd_rob' + tg] = [o[0] for o in out]; Fv['corr_len_rob' + tg] = [o[1] for o in out]
Fv['low_signal'] = (Fv.sbr488 < LOW_SIGNAL_SBR) | (Fv.sbr405 < LOW_SIGNAL_SBR)
print("low-signal spheroids:", Fv[Fv.low_signal][['timepoint','well','condition','sbr488','sbr405']].round(2).to_string(index=False))
Fv.to_csv('spatial_features_v2.csv', index=False)

# ---- shape (roundness) QC: robust outlier score against the vehicle, plus an absolute roundness gate
if SHAPE_QC_CFG:
    _ref = Fv[Fv.condition == SHAPE_QC_CFG.get('z_ref_condition', 'DMSO')]
    for _c, _sgn in [('circularity', -1), ('solidity', -1), ('edge_roughness', 1)]:
        _m = float(np.median(_ref[_c])); _s = float(1.4826 * np.median(np.abs(_ref[_c] - _m)))
        Fv['z_' + _c] = _sgn * (Fv[_c] - _m) / _s if _s > 0 else np.nan
    Fv['shape_z'] = Fv[['z_circularity', 'z_solidity', 'z_edge_roughness']].max(axis=1)
    Fv['not_round'] = ((Fv.circularity < SHAPE_QC_CFG.get('circularity_min', 0)) |
                       (Fv.solidity < SHAPE_QC_CFG.get('solidity_min', 0)) |
                       (Fv.edge_roughness > SHAPE_QC_CFG.get('edge_roughness_max', np.inf)))
    print('not round:', int(Fv.not_round.sum()), 'of', len(Fv),
          '| by condition:', Fv[Fv.not_round].condition.value_counts().to_dict())
else:
    Fv['not_round'] = False

VIS = set(map(tuple, VIS_DEBRIS))
Fv['qc_exclude'] = Fv.low_signal | np.array([k in VIS for k in zip(Fv.timepoint, Fv.well)], bool)
if SHAPE_QC_CFG and SHAPE_QC_CFG.get('exclude', True):
    Fv['qc_exclude'] = Fv.qc_exclude | Fv.not_round
Fv.to_csv('spatial_features_v2.csv', index=False)
pickle.dump({k: v['_cal'] for k, v in MAPS.items()}, open('spatial_maps_cal.pkl', 'wb'), protocol=4)
pickle.dump(PROF, open('radial_profiles.pkl', 'wb'), protocol=4)
print(Fv.shape, 'low_signal:', int(Fv.low_signal.sum()), 'qc_exclude:', int(Fv.qc_exclude.sum()))
print(f'step 4 done in {time.time()-T0:.0f}s')
