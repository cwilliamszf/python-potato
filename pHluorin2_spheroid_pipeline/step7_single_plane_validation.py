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

"""Step 7 - single-plane validation of the spatial findings against depth / optical artefacts.
Uses the raw z-stacks: equatorial plane (sharpest 488 outline), chromatic shift (405 focuses ~2 slices higher),
chromatic-matched single-plane features vs max projection, depth series, single-plane patch tracking, and the
405-channel medium signal (OGM fluorescence). Outputs: zplane_*.csv, medium_background_by_well.csv,
optical_validation.svg, maxproj_vs_plane_maps.svg"""

run_lib('common.py'); run_lib('stage1_extract.py'); run_lib('stage2_features.py')
with np.load('stage1_arrays.npz') as Z:
    keys = [tuple(k.split('|')) for k in Z['keys']]; gA = Z['g']; rA = Z['r']; mA = Z['m']
s1 = pd.read_csv('stage1_status.csv')
okset = set(zip(s1[s1.status == 'ok'].timepoint, s1[s1.status == 'ok'].well))
G = {k: gA[i] for i, k in enumerate(keys)}; Rr = {k: rA[i] for i, k in enumerate(keys)}
M = {k: mA[i] for i, k in enumerate(keys) if k in okset}
EXCL = {(t, w) for t in TPS for w in EXCLUDE_WELLS_ALL_TP}
keys_ok = [k for k in M if k not in EXCL]

run_lib('zstack_extract.py'); run_lib('zstack_analysis.py')
# FLOORS and ALPHA come from lib/zstack_analysis.py: the step-4 calibration values rounded as in the original run
# (alpha 0.32/0.32/0.22). To use this run's exact step-4 values instead, uncomment:
# cal = json.load(open('calibration.json')); FLOORS = {k: tuple(v) for k, v in cal['floors'].items()}; ALPHA = cal['alpha']
fields, med = build_fields(G, Rr, M, keys_ok, FLOORS)
for tp in TPS: fields[tp]['alpha'] = ALPHA[tp]
del gA, rA; gc.collect()
run_lib('zfull.py')
print('== per-plane extraction for every spheroid (slow: ~6 s per spheroid)', flush=True)
RI = {tp: raw_index(PLATES[tp]) for tp in TPS}
if os.path.exists('zplane_run_all.pkl'):
    ZF, ZPROF, ZMAPS = pickle.load(open('zplane_run_all.pkl', 'rb'))
else:
    if os.path.exists('zplane_features.csv'): os.remove('zplane_features.csv')   # run_all resumes from this file; start clean
    ZF, ZPROF, ZMAPS = run_all(RI)
    pickle.dump((ZF, ZPROF, ZMAPS), open('zplane_run_all.pkl', 'wb'), protocol=4)   # cache: delete to recompute
pickle.dump({'prof': ZPROF}, open('zplane_profiles.pkl', 'wb'), protocol=4)
SF = pd.read_csv('spatial_features_v2.csv')
print('== comparison with max projection', flush=True)

exec(open(LIB('zcompare.py'), encoding='utf-8').read())
q = SF[['timepoint', 'well', 'qc_exclude', 'low_signal']]
ZQ = ZF.merge(q, on=['timepoint', 'well'], how='left')
ZQ = ZQ[~ZQ.qc_exclude.fillna(False).astype(bool) & (ZQ.condition != 'Fluorobrite')]
ZQ.to_csv('zplane_features_qc.csv', index=False)
print("n spheroids:", len(ZQ), "| z_eq (slices) median/IQR:", ZQ.z_eq.median(), ZQ.z_eq.quantile([.25, .75]).tolist(),
      "| R_um median:", round(ZQ.R_um.median()))
print("max-proj source plane 405 (median slice): rim", ZQ.mp_src405_rim.median(), "core", ZQ.mp_src405_core.median(),
      "| 488 rim/core:", ZQ.mp_src488_rim.median(), ZQ.mp_src488_core.median())
E = effects(ZQ); E.to_csv('zplane_condition_effects.csv', index=False)
A1 = agreement(E, 'mp_', 'eq_').assign(comparison='max-proj vs matched plane')
A2 = agreement(E, 'eq_', 'equ_').assign(comparison='matched vs unmatched plane')
A = pd.concat([A1, A2]); A.to_csv('zplane_agreement.csv', index=False)
print(A.round(2).to_string(index=False))
# per-spheroid agreement
print({f: round(stats.spearmanr(ZQ['mp_' + f], ZQ['eq_' + f], nan_policy='omit')[0], 2) for f in ZFEAT})


e = E[['timepoint', 'condition', 'mp_ratio_median', 'eq_ratio_median', 'mp_ratio_median_es', 'eq_ratio_median_es']].copy()
e['disagree'] = np.sign(e.mp_ratio_median) != np.sign(e.eq_ratio_median)
print(e[e.disagree | (e.mp_ratio_median.abs() > 0.2)].sort_values(['condition', 'timepoint']).round(3).to_string(index=False))
# depth series aligned on equatorial plane: height h = (z - z_eq) * 10.1 um ; keep h in [0, 180] um (inside upper hemisphere)
H = np.arange(0, 19)  # slices above equator
dep = []
for (t, w), p in ZPROF.items():
    r = ZQ[(ZQ.timepoint == t) & (ZQ.well == w)]
    if r.empty: continue
    ze = int(r.z_eq.iloc[0])
    for reg in ['rim', 'core', 'whole']:
        v = p[f'm_{reg}']
        for h in H:
            z = ze + h
            if z < len(v) and np.isfinite(v[z]): dep.append(dict(timepoint=t, well=w, condition=COND[w], region=reg, h_slices=h, h_um=h * 10.1, ratio=v[z]))
DEP = pd.DataFrame(dep); DEP.to_csv('zplane_depth_series.csv', index=False)
dm = DEP[DEP.condition == 'DMSO'].groupby(['timepoint', 'region', 'h_slices']).ratio.mean().rename('dmso')
DEP = DEP.join(dm, on=['timepoint', 'region', 'h_slices']); DEP['log_contrast'] = np.log(DEP.ratio / DEP.dmso)
print("DMSO whole-spheroid ratio by height (48h):", DEP[(DEP.condition=='DMSO')&(DEP.region=='whole')&(DEP.timepoint=='48h')].groupby('h_slices').ratio.mean().iloc[[0,3,6,9,12,15,18]].round(2).to_dict())
sel = ['Baf L', 'Olig M', '3BP H', 'Rot H', 'Oz L', '20uM OGM', '5uM OGM', '50uM Ogerin', 'Phlor H']
Cdep = DEP[DEP.condition.isin(sel) & (DEP.region == 'whole') & (DEP.timepoint == '72h')].groupby(['condition', 'h_slices']).log_contrast.mean().unstack()
print("72h whole-spheroid log(ratio/DMSO) at heights 0,3,6,9,12,15,18 slices:"); print(np.exp(Cdep[[0, 3, 6, 9, 12, 15, 18]]).round(2).to_string())


rc = DEP[DEP.region.isin(['rim', 'core'])].pivot_table(index=['timepoint', 'well', 'condition', 'h_slices'], columns='region', values='ratio').reset_index()
rc['rim_core'] = rc.rim - rc.core
RC = rc[rc.timepoint == '72h'].groupby(['condition', 'h_slices']).rim_core.mean().unstack()
print("72h rim − core by height (slices above equator):"); print(RC.loc[['DMSO'] + sel, [0, 3, 6, 9, 12, 15]].round(2).to_string())
rc.to_csv('zplane_rim_core_by_height.csv', index=False)
# patch persistence in single plane
exec(open(LIB('tracking.py'), encoding='utf-8').read())
ZM = {k: {'eq': v} for k, v in ZMAPS.items()}
GK = set(zip(ZQ.timepoint, ZQ.well))
TSz, TNz = tracking_tables(ZM, GK, 'eq', COND, null_per_pair=300)
TSz.to_csv('zplane_tracking_same.csv', index=False); TNz.to_csv('zplane_tracking_null.csv', index=False)
for p in ['24h->48h', '48h->72h', '24h->72h']:
    a, b = TSz[TSz.pair == p].r_fixed.dropna(), TNz[TNz.pair == p].r_fixed.dropna()
    print(p, 'same', round(a.median(), 3), 'null', round(b.median(), 3), 'p', f"{stats.mannwhitneyu(a, b, alternative='greater')[1]:.1e}", len(a))


MPM = {}
jobs = []
for k in sorted(GK):
    m = M[k]; far = ~fo.dilate_iter(m, 20)                 # == ~ndi.binary_dilation(m, iterations=20)
    g = G[k].astype(np.float32); r = Rr[k].astype(np.float32)
    jobs.append((k, g, r, m, (np.median(g[far]), np.median(r[far])), 0))
from parallel import WellPool
with WellPool(HERE, ['stage2_features.py'], shared={'PAR_FIELDS': fields}, fallback_ns=globals()) as pool:
    for job, mp_cal in zip(jobs, pool.imap('well_features_job', jobs, chunksize=2)):
        MPM[job[0]] = mp_cal
del jobs
rows = []
kl = sorted(GK); rng = np.random.default_rng(3)
for k in kl:
    r0, _, _ = pair_corr(MPM[k], ZMAPS[k], rot_search=False)
    other = kl[rng.integers(len(kl))]
    while other == k or other[0] != k[0]: other = kl[rng.integers(len(kl))]
    rn, _, _ = pair_corr(MPM[k], ZMAPS[other], rot_search=False)
    rows.append(dict(timepoint=k[0], well=k[1], condition=COND[k[1]], r_same=r0, r_other=rn))
PM = pd.DataFrame(rows); PM.to_csv('zplane_vs_maxproj_patchmap_corr.csv', index=False)
print(PM.groupby('timepoint')[['r_same', 'r_other']].median().round(2).to_string(), "\nMWU p:", f"{stats.mannwhitneyu(PM.r_same, PM.r_other, alternative='greater')[1]:.1e}")
print("DMSO rim-core: max-proj", ZQ[ZQ.condition=='DMSO'].groupby('timepoint').mp_rim_minus_core.mean().round(3).to_dict(),
      "| matched plane", ZQ[ZQ.condition=='DMSO'].groupby('timepoint').eq_rim_minus_core.mean().round(3).to_dict())

print('== medium 405 signal (compound fluorescence check)', flush=True)

rows = []
for k in keys:
    t, w = k
    g = G[k].astype(np.float32); r = Rr[k].astype(np.float32)
    m = M.get(k, np.zeros((490, 490), bool))
    far = ~fo.dilate_iter(m, 20)                           # == ~ndi.binary_dilation(m, iterations=20)
    rows.append(dict(timepoint=t, well=w, row=w[0], col=int(w[1:]), condition=COND[w],
                     g_med=float(np.median(g[far])) - FLOORS[t][0], r_med=float(np.median(r[far])) - FLOORS[t][1]))
MED = pd.DataFrame(rows); MED['med_ratio'] = MED.g_med / MED.r_med
# position model on non-OGM, non-Ogerin wells: row + col effects, then residual by condition
base = MED[~MED.condition.str.contains('OGM|Ogerin')]
import statsmodels.formula.api as smf
fit = smf.ols('np.log(r_med) ~ C(timepoint)*C(row) + C(col)', MED[MED.col <= 9]).fit()
MED['r_med_pred'] = np.exp(fit.predict(MED.assign(col=np.where(MED.col > 9, 9, MED.col))))
MED['r_excess'] = MED.r_med / MED.r_med_pred
print(MED[MED.col >= 10].groupby(['condition', 'timepoint']).r_excess.mean().unstack().round(2)
      .loc[['0.5uM OGM','1uM OGM','5uM OGM','20uM OGM','0.5uM Ogerin','5uM Ogerin','20uM Ogerin','50uM Ogerin']].to_string())
print("cols 1-9 residual SD of log r_med:", round(np.std(np.log(MED[MED.col <= 9].r_excess)), 3))
print(MED.groupby(['condition'])[['g_med','r_med']].median().loc[['DMSO','Fluorobrite','Baf L','Phlor H','0.5uM OGM','5uM OGM','20uM OGM','50uM Ogerin']].round(0).to_string())
MED.to_csv('medium_background_by_well.csv', index=False)


SF['g_sig'] = SF.g_mean - SF.timepoint.map(lambda t: FLOORS[t][0]); SF['r_sig'] = SF.r_mean - SF.timepoint.map(lambda t: FLOORS[t][1])
SF = SF.merge(MED[['timepoint', 'well', 'r_excess', 'g_med', 'r_med']], on=['timepoint', 'well'])
ctl = SF[SF.condition.isin(['DMSO', 'Fluorobrite', '0.5uM Ogerin', '5uM Ogerin', '20uM Ogerin'])].groupby('timepoint')[['g_sig', 'r_sig']].mean()
o = SF[SF.condition.str.contains('OGM')].copy()
o['g_rel'] = o.g_sig / o.timepoint.map(ctl.g_sig); o['r_rel'] = o.r_sig / o.timepoint.map(ctl.r_sig)
print(o.groupby(['condition', 'timepoint'])[['g_rel', 'r_rel', 'r_excess', 'ratio_median_cal']].mean().round(2).to_string())
print("rho(spheroid 405 rel, medium 405 excess) across OGM wells:", round(stats.spearmanr(o.r_rel, o.r_excess)[0], 2),
      "| rho(ratio_cal, medium excess):", round(stats.spearmanr(o.ratio_median_cal, o.r_excess)[0], 2))

run_lib('figstyle.py'); run_lib('fig_optical_validation.py')
run_lib('fig_maxproj_vs_plane.py')
print(f'step 7 done in {time.time()-T0:.0f}s')