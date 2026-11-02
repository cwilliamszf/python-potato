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

"""Step 5 - statistics on per-spheroid spatial features (calibrated background model, suffix _cal).
Outputs: dose_time_models.csv, tracking_same_spheroid.csv, tracking_null_pairs.csv, radial_excess_profiles.csv,
background_model_sensitivity.csv, covariate_adjusted_effects.csv, matched_mean_*.csv, matched_mean_family_tests.csv,
feature_icc.csv, trajectory_classification.csv"""
run_lib('common.py'); run_lib('stage2_features.py'); run_lib('analysis_stats.py'); run_lib('tracking.py')
Fv = pd.read_csv('spatial_features_v2.csv')
MAPS = {k: {'_cal': v} for k, v in pickle.load(open('spatial_maps_cal.pkl', 'rb')).items()}
PROF = pickle.load(open('radial_profiles.pkl', 'rb'))
print('== dose x time models', flush=True)
FEAT = {'ratio_median_cal': 'mean acidity (median 488/405)', 'rim_minus_core_cal': 'rim − centre 488/405',
        'resid_sd_rob_cal': 'residual SD (robust, detrended)', 'acid_patch_frac_cal': 'acidic-patch area fraction',
        'corr_len_rob_cal': 'correlation length (full-res px)', 'patch_boundary_enrich_cal': 'patch boundary enrichment (outer 20 %)',
        'area_px': 'spheroid area (full-res px²)'}
Fq = Fv[(Fv.condition != 'Fluorobrite') & ~Fv.low_signal].copy()
DT = pd.DataFrame([dose_time_model(Fq, c, y) for c in COMPOUNDS for y in FEAT])
for tp in TPS: DT[f'q_slope_{tp}'] = bh(DT[f'p_slope_{tp}'])
DT['q_dose_x_time'] = bh(DT['p_dose_x_time']); DT.to_csv('dose_time_models.csv', index=False)

good = Fv[~Fv.qc_exclude]; GOODK = set(zip(good.timepoint, good.well))
print('== spheroid tracking across days', flush=True)
TS, TN = tracking_tables(MAPS, GOODK, '_cal', COND, null_per_pair=300)
TS.to_csv('tracking_same_spheroid.csv', index=False); TN.to_csv('tracking_null_pairs.csv', index=False)
for col in ['r_fixed', 'r_rotmax']:
    print(col, 'same', TS.groupby('pair')[col].median().round(3).to_dict(), '| null', TN.groupby('pair')[col].median().round(3).to_dict())

def interior_corr(mpA, mpB, cut=0.35):
    A = norm_disk(mpA['res'], mpA['centroid'], mpA['eq_r']); B = norm_disk(mpB['res'], mpB['centroid'], mpB['eq_r'])
    inner = (UU**2 + VV**2) <= (1 - cut)**2 * 0.92**2
    return _r(np.where(inner, A, np.nan), np.where(inner, B, np.nan))
def shape_corr(kA, kB):
    a = MAPS[kA]['_cal']; b = MAPS[kB]['_cal']
    A = norm_disk(np.isfinite(a['R'].astype(np.float32)).astype(np.float32), a['centroid'], a['eq_r'])
    B = norm_disk(np.isfinite(b['R'].astype(np.float32)).astype(np.float32), b['centroid'], b['eq_r'])
    return _r(np.nan_to_num(A), np.nan_to_num(B))
TS['r_interior'] = [interior_corr(MAPS[(p.split('->')[0], w)]['_cal'], MAPS[(p.split('->')[1], w)]['_cal']) for p, w in zip(TS.pair, TS.well)]
TS['r_shape'] = [shape_corr((p.split('->')[0], w), (p.split('->')[1], w)) for p, w in zip(TS.pair, TS.well)]
sub = TN.groupby('pair').head(120).copy()
sub['r_interior'] = [interior_corr(MAPS[(p.split('->')[0], a)]['_cal'], MAPS[(p.split('->')[1], b)]['_cal']) for p, a, b in zip(sub.pair, sub.well_a, sub.well_b)]
print("interior-only r: same", TS.groupby('pair').r_interior.median().round(3).to_dict(), "null", sub.groupby('pair').r_interior.median().round(3).to_dict())
print("spearman(r_fixed, r_shape):", round(stats.spearmanr(TS.r_fixed, TS.r_shape, nan_policy='omit')[0], 2))
TS['compound'] = [cond_info(c)[0] for c in TS.condition]
pc = TS[TS.pair==LASTPAIR].groupby('condition')[['r_fixed', 'r_interior']].median()
print(pc.loc[[c for c in display_order if c in pc.index]].round(2).T.to_string())
TS.to_csv('tracking_same_spheroid.csv', index=False)

print('== radial location of excess acidity', flush=True)

print([c for c in good.columns if 'patch' in c and c.endswith('_cal')], [c for c in good.columns if 'depth' in c and c.endswith('_cal')])
# excess-acidity radial profile: per-spheroid normalised-depth profile minus same-plate DMSO mean profile
ndc = (np.linspace(0, 1, 11)[:-1] + np.linspace(0, 1, 11)[1:]) / 2
P = {k: v for k, v in PROF['_prof_nd_cal'].items() if k in GOODK}
excess = []
for tp in TPS:
    dm = np.nanmean([P[(tp, w)] for w in WELLS if COND[w] == 'DMSO' and (tp, w) in P], 0)
    for w in WELLS:
        if (tp, w) in P:
            e = P[(tp, w)] - dm
            excess.append(dict(timepoint=tp, well=w, condition=COND[w], **{f'nd{i}': e[i] for i in range(10)}))
EX = pd.DataFrame(excess)
EX['edge_excess'] = EX[['nd0', 'nd1']].mean(1); EX['inner_excess'] = EX[['nd6','nd7','nd8','nd9']].mean(1)
EX['edge_share'] = EX.edge_excess - EX.inner_excess
EX.to_csv('radial_excess_profiles.csv', index=False)
q = EX.groupby(['condition', 'timepoint'])[['edge_excess', 'inner_excess', 'edge_share']].mean().unstack('timepoint')
q = q.loc[[c for c in display_order if c in q.index]]
print(q['edge_share'].round(2).T.to_string())

print('== optical / geometric controls', flush=True)

g2 = good.copy(); g2['lsbr'] = np.log(np.sqrt(g2.sbr488 * g2.sbr405)); g2['larea'] = np.log(g2.area_px)
ctl = {}
# 1. dependence of ratio on spheroid brightness (signal over medium), raw vs calibrated, all spheroids, within timepoint
for col in ['ratio_median', 'ratio_median_fs', 'ratio_median_cal']:
    ctl[f'rho_{col}_vs_SBR'] = {tp: round(stats.spearmanr(d.lsbr, d[col])[0], 2) for tp, d in g2.groupby('timepoint')}
# within near-null conditions only (no expected pH change)
nul = g2[g2.condition.isin(NULL_CONDS_CFG or ['DMSO','Fluorobrite','Phlor L','Phlor M','Phlor H','0.5uM OGM','0.5uM Ogerin','Ace H','Rot L'])]
for col in ['ratio_median', 'ratio_median_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal', 'rim_minus_core_cal']:
    ctl[f'null_rho_{col}_vs_SBR'] = round(stats.spearmanr(nul.lsbr, nul[col])[0], 2)
    ctl[f'null_rho_{col}_vs_area'] = round(stats.spearmanr(nul.larea, nul[col])[0], 2)
for k, v in ctl.items(): print(k, v)
# 2. sign agreement of condition effects across 4 background models (72h and 48h)
mods = ['', '_fs', '_cal', '_ms']
agree = []
for tp in LATE:
    d = good[good.timepoint == tp]; dm = d[d.condition == 'DMSO']
    for c in d.condition.unique():
        if c == 'DMSO': continue
        x = d[d.condition == c]
        for base in ['ratio_median', 'acid_patch_frac', 'rim_minus_core']:
            v = [x[base + m].mean() - dm[base + m].mean() for m in mods]
            agree.append(dict(timepoint=tp, condition=c, feature=base, **{f'd{m or "_raw"}': vv for m, vv in zip(mods, v)},
                              sign_agree=int(np.nansum(np.sign(v) == np.sign(v[2]))) ))
AG = pd.DataFrame(agree); AG.to_csv('background_model_sensitivity.csv', index=False)
print(AG.groupby('feature').sign_agree.value_counts().unstack().fillna(0).astype(int).to_string())
# 3. brightness vs spectral within spheroid
print("corr(spectral, brightness) residuals, median by tp:", good.groupby('timepoint').corr_spec_bright_cal.median().round(2).to_dict())
print("patch brightness residual (log) median:", round(good.patch_bright_resid_cal.median(), 3), "IQR", good.patch_bright_resid_cal.quantile([.25,.75]).round(3).tolist())


g2 = g2[g2.condition != 'Fluorobrite'].copy()
for col in ['ratio_median_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal', 'rim_minus_core_cal']:
    dd = g2.copy(); grp_ = dd.groupby(['condition', 'timepoint'])
    a = dd[col] - grp_[col].transform('mean'); b = dd.lsbr - grp_.lsbr.transform('mean'); c_ = dd.larea - grp_.larea.transform('mean')
    print(col, 'within-cond rho vs SBR', round(stats.spearmanr(a, b)[0], 2), '| vs area', round(stats.spearmanr(a, c_)[0], 2))
adj_rows = []
for col in ['ratio_median_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal', 'rim_minus_core_cal', 'corr_len_rob_cal', 'patch_boundary_enrich_cal']:
    for tp in TPS:
        d = g2[g2.timepoint == tp].dropna(subset=[col])
        f0 = smf.ols(f'{col} ~ C(condition, Treatment("DMSO"))', d).fit(cov_type='HC3')
        f1 = smf.ols(f'{col} ~ C(condition, Treatment("DMSO")) + lsbr + larea', d).fit(cov_type='HC3')
        for nm in f0.params.index:
            if 'condition' not in nm: continue
            adj_rows.append(dict(feature=col, timepoint=tp, condition=nm.split('[T.')[1][:-1], unadj=f0.params[nm], adj=f1.params[nm],
                                 p_adj=f1.pvalues[nm], b_lsbr=f1.params['lsbr'], b_larea=f1.params['larea']))
ADJ = pd.DataFrame(adj_rows); ADJ['q_adj'] = np.nan
for (f, t), idx in ADJ.groupby(['feature', 'timepoint']).groups.items(): ADJ.loc[idx, 'q_adj'] = bh(ADJ.loc[idx, 'p_adj'])
ADJ.to_csv('covariate_adjusted_effects.csv', index=False)
x = ADJ[(ADJ.timepoint == LAST) & ADJ.feature.isin(['ratio_median_cal', 'acid_patch_frac_cal', 'resid_sd_rob_cal'])]
print(x.pivot(index='condition', columns='feature', values=['unadj', 'adj']).loc[[c for c in display_order if c in set(x.condition)]].round(3).to_string())
print("slope agreement (adj/unadj, 72h, |unadj|>small):", ADJ[(ADJ.timepoint==LAST)].groupby('feature').apply(lambda d: round(np.median((d.adj/d.unadj)[d.unadj.abs() > d.unadj.abs().median()]), 2)).to_dict())

print('== matched-mean comparison', flush=True)

gq = good[good.condition != 'Fluorobrite'].copy()
MM = {}
for het in ['resid_sd_rob_cal', 'acid_patch_frac_cal', 'corr_len_rob_cal']:
    P_, S_ = matched_mean(gq, het, k=5); MM[het] = (P_, S_)
    P_.to_csv(f'matched_mean_{het}.csv', index=False)

def perm_family(P_, cpd, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = P_[P_.timepoint.isin(LATE)]
    obs = d[d.compound == cpd].delta.mean(); null = []
    for _ in range(n):
        lab = rng.permutation(d.compound.values); null.append(d.delta.values[lab == cpd].mean())
    null = np.array(null); return obs, float((np.abs(null) >= abs(obs)).mean())
PF = pd.DataFrame([dict(compound=c, feature=h, **dict(zip(['delta','p_perm'], perm_family(MM[h][0], c))))
                   for h in MM for c in (PERM_FAMILY_CFG or ['DMSO'] + COMPOUNDS + ['TMZ'])])
PF['q'] = bh(PF.p_perm); PF.to_csv('matched_mean_family_tests.csv', index=False)
print(PF.pivot(index='compound', columns='feature', values=['delta','q']).round(3).to_string())

print('== reproducibility (ICC, leave-one-spheroid-out trajectories)', flush=True)

ICC = pd.DataFrame({f: icc1(gq, f) for f in FEAT}).T
print(ICC.round(2).to_string())
traj_feats = {'mean only': ['ratio_median_cal'], 'heterogeneity only': ['resid_sd_rob_cal', 'acid_patch_frac_cal', 'corr_len_rob_cal'],
              'mean + heterogeneity': ['ratio_median_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal', 'corr_len_rob_cal'],
              'mean + heterogeneity + radial': ['ratio_median_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal', 'corr_len_rob_cal', 'rim_minus_core_cal']}
gq['cpd_dose'] = gq.condition
gq['compound2'] = gq.compound
# label-permutation null: the 200 shuffles per model are drawn here in the original order (same seeds), then all
# 8 x 200 leave-one-out classifications run in parallel worker processes (lib/parallel.py)
from parallel import WellPool
rows, null_jobs, specs = [], [], []
for nm, fs in traj_feats.items():
    for lab in ['cpd_dose', 'compound2']:
        cols = ['well', 'timepoint', lab] + fs          # all loo_classify uses
        rng = np.random.default_rng(0)
        for _ in range(200):
            gp = gq[cols].copy(); wl = gp.drop_duplicates('well')[['well', lab]]
            perm = dict(zip(wl.well, rng.permutation(wl[lab].values))); gp[lab] = gp.well.map(perm)
            null_jobs.append((gp, fs, lab))
        specs.append((nm, fs, lab))
with WellPool(HERE, ['analysis_stats.py'], fallback_ns=globals()) as pool:
    null_acc = [r[0] for r in pool.imap('loo_classify', null_jobs, chunksize=8)]
for i, (nm, fs, lab) in enumerate(specs):
    acc, ch, n = loo_classify(gq, fs, label=lab)
    nulls = null_acc[200 * i: 200 * (i + 1)]
    rows.append(dict(features=nm, label={'cpd_dose':'condition (compound × dose)','compound2':'compound'}[lab], accuracy=acc,
                     chance=ch, null_95=float(np.percentile(nulls, 95)), p_perm=float((np.array(nulls) >= acc).mean()), n_spheroids=n))
LOO = pd.DataFrame(rows); LOO.to_csv('trajectory_classification.csv', index=False); ICC.to_csv('feature_icc.csv')
print(LOO.round(3).to_string(index=False))

print(f'step 5 done in {time.time()-T0:.0f}s')
