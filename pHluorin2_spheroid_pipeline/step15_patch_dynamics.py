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
"""Step 15 - acid-patch dynamics across timepoints (same well = same spheroid), from the step-4 patch maps;
alkaline patches (log residual < -log 1.06) are analysed the same way (outputs *_alk.csv / *_alk.svg).
Phenotypes: (1) patch movement - centroid displacement of linked patches, mask Jaccard distance and centroid-set
Jaccard distance; (2) patch growth relative to the spheroid; (3) number of patches and emergence / loss (new,
vanished, split, merged); (4) acidity of the patches themselves (calibrated 488/405 inside each patch, contrast
with the surrounding and the same-depth tissue) and a per-patch brightness check for signal-loss artefacts.
Methods: lib/patch_dynamics.py. Every metric is also computed for random pairs of
DIFFERENT spheroids at the same timepoints (chance level). Works on any results folder (max projection or
equatorial plane). Inputs: spatial_features_v2.csv, spatial_maps_cal.pkl, stage1_arrays.npz.
Outputs: patch_table.csv, patch_per_spheroid.csv, patch_transitions.csv, patch_links.csv, patch_transitions_null.csv,
patch_condition_summary.csv, patch_dose_models.csv, patch_dynamics_tracks.svg, patch_dynamics_summary.svg,
patch_acidity.svg"""
run_lib('common.py'); run_lib('stage2_features.py'); run_lib('analysis_stats.py'); run_lib('patch_dynamics.py')
Fv = pd.read_csv('spatial_features_v2.csv')
MAPS = pickle.load(open('spatial_maps_cal.pkl', 'rb'))
GOOD = set(zip(Fv[~Fv.qc_exclude].timepoint, Fv[~Fv.qc_exclude].well))
INFO = {c: cond_info(c) for c in set(COND.values())}
N_NULL = int(os.environ.get('SPHEROID_PATCH_NULL', 200))
with np.load('stage1_arrays.npz') as Z:          # 488 / 405 arrays for the per-patch brightness check
    _ix = {tuple(k.split('|')): i for i, k in enumerate(Z['keys'])}; GA, RA = Z['g'], Z['r']
MED = dict(zip(zip(Fv.timepoint, Fv.well), zip(Fv.g_medium, Fv.r_medium)))

POLARITIES = [x.strip() for x in os.environ.get('SPHEROID_PATCH_POLARITIES', 'acid,alk').split(',') if x.strip()]
for POL in POLARITIES:      # acid patches -> patch_*.csv ; alkaline patches -> patch_*_alk.csv
    SUF = '' if POL == 'acid' else '_alk'
    print(f'==== {POL} patches ====', flush=True)
    print('== patches per spheroid-timepoint', flush=True)
    PT, LAB, SUMM, prow = {}, {}, [], []
    for tp in TPS:
        for w in WELLS:
            k = (tp, w)
            if k not in MAPS: continue
            T, s, L = spheroid_patches(MAPS[k], GA[_ix[k]], RA[_ix[k]], MED[k], POL) if k in MED else spheroid_patches(MAPS[k], polarity=POL)
            PT[k], LAB[k] = T, L
            cpd, dum, drk = INFO[COND[w]]
            SUMM.append(dict(timepoint=tp, well=w, condition=COND[w], compound=cpd, dose_um=dum, dose_rank=drk,
                             qc_exclude=k not in GOOD, **s))
            prow += [dict(timepoint=tp, well=w, condition=COND[w], **r) for r in T.to_dict('records')]
    SP = pd.DataFrame(SUMM); SP.to_csv(f'patch_per_spheroid{SUF}.csv', index=False)
    pd.DataFrame(prow).to_csv(f'patch_table{SUF}.csv', index=False)
    print(SP[~SP.qc_exclude].groupby('timepoint')[['n_patches', 'patch_frac', 'mean_patch_area_um2', 'patch_ratio_mean',
          'nonpatch_ratio_median', 'patch_contrast_depth_mean', 'frac_signal_loss']].median().round(3).to_string())

    PAIRS = list(zip(TPS[:-1], TPS[1:])) + ([(TPS[0], TPS[-1])] if len(TPS) > 2 else [])
    if os.environ.get('SPHEROID_PATCH_FIGURES_ONLY'):      # redraw the figures from the saved tables
        TR = pd.read_csv(f'patch_transitions{SUF}.csv'); NU = pd.read_csv(f'patch_transitions_null{SUF}.csv')
        links = pd.read_csv(f'patch_links{SUF}.csv').to_dict('records')
    else:
        print('== transitions (same spheroid)', flush=True)
        REG = ['fixed', 'rotreg']
        rows, links = [], []
        for t1, t2 in PAIRS:
            for w in WELLS:
                a, b = (t1, w), (t2, w)
                if a not in GOOD or b not in GOOD or a not in MAPS or b not in MAPS: continue
                gA = resample(MAPS[a], LAB[a])
                cpd, dum, drk = INFO[COND[w]]
                for reg in REG:
                    o, Lk = transition(MAPS[a], PT[a], dict(SP[(SP.timepoint == t1) & (SP.well == w)].iloc[0]), LAB[a],
                                       MAPS[b], PT[b], dict(SP[(SP.timepoint == t2) & (SP.well == w)].iloc[0]), LAB[b], reg, gA)
                    meta = dict(pair=f'{t1}->{t2}', dt_h=HOURS[t2] - HOURS[t1], register=reg, well=w, condition=COND[w],
                                compound=cpd, dose_um=dum, dose_rank=drk)
                    rows.append(dict(meta, **o)); links += [dict(meta, **r) for r in Lk.to_dict('records')]
        TR = per_day(pd.DataFrame(rows)); TR.to_csv(f'patch_transitions{SUF}.csv', index=False)
        pd.DataFrame(links).to_csv(f'patch_links{SUF}.csv', index=False)

        print('== chance level: random pairs of different spheroids', flush=True)
        rng = np.random.default_rng(0); nrows = []
        SPi = SP.set_index(['timepoint', 'well'])
        for t1, t2 in PAIRS:
            w1 = [w for w in WELLS if (t1, w) in GOOD and (t1, w) in MAPS]; w2 = [w for w in WELLS if (t2, w) in GOOD and (t2, w) in MAPS]
            allp = [(x, y) for x in w1 for y in w2 if x != y]
            for ii in rng.choice(len(allp), size=min(N_NULL, len(allp)), replace=False):
                x, y = allp[ii]; a, b = (t1, x), (t2, y); gA = resample(MAPS[a], LAB[a])
                for reg in REG:
                    o, _ = transition(MAPS[a], PT[a], dict(SPi.loc[a]), LAB[a], MAPS[b], PT[b], dict(SPi.loc[b]), LAB[b], reg, gA)
                    nrows.append(dict(pair=f'{t1}->{t2}', dt_h=HOURS[t2] - HOURS[t1], register=reg, well_a=x, well_b=y, **o))
        NU = per_day(pd.DataFrame(nrows)); NU.to_csv(f'patch_transitions_null{SUF}.csv', index=False)

        METRICS = ['jaccard_mask_dist', 'centroid_jaccard_dist', 'disp_per_day', 'disp_um_per_day', 'radial_shift_mean',
                   'rel_growth_patch_per_day', 'rel_growth_total_per_day', 'emerged_per_day', 'vanished_per_day', 'n_split',
                   'n_merged', 'turnover', 'contrast_change_mean', 'ratio_log2fc_mean', 'rotation_deg']
        CS = TR.groupby(['register', 'pair', 'condition'])[METRICS].mean().reset_index()
        CS = CS.merge(TR.groupby(['register', 'pair', 'condition']).size().rename('n').reset_index())
        nul = NU.groupby(['register', 'pair'])[[m for m in METRICS if m in NU]].median().add_suffix('_null').reset_index()
        CS = CS.merge(nul, on=['register', 'pair'], how='left'); CS.to_csv(f'patch_condition_summary{SUF}.csv', index=False)
        print('chance level (median of random pairs):'); print(nul.round(3).to_string(index=False))

        print('== dose models', flush=True)
        Fq = SP[~SP.qc_exclude & (SP.condition != 'Fluorobrite')].copy()
        DT = pd.DataFrame([dict(level='per timepoint', **dose_time_model(Fq, c, y)) for c in COMPOUNDS
                           for y in ['n_patches', 'patch_density_per_mm2', 'patch_frac', 'mean_patch_area_um2', 'patch_ratio_mean',
                                 'patch_contrast_depth_mean', 'frac_signal_loss']])
        dm = []
        TRf = TR[TR.condition != 'Fluorobrite']
        for reg in REG:
            for pr in TRf.pair.unique():
                for c in COMPOUNDS:
                    d0 = TRf[(TRf.register == reg) & (TRf.pair == pr) & TRf.compound.isin(['DMSO', c])]
                    for y in METRICS:
                        d = d0.dropna(subset=[y])
                        if d.compound.nunique() < 2 or d[y].nunique() < 2: continue
                        f = smf.ols(f'{y} ~ dose_rank', d).fit(cov_type='HC3')
                        dm.append(dict(level='transition', register=reg, pair=pr, compound=c, feature=y, n_spheroids=len(d),
                                       dmso_mean=d[d.compound == 'DMSO'][y].mean(), slope_per_dose_rank=f.params['dose_rank'],
                                       p_slope=f.pvalues['dose_rank'], rho=stats.spearmanr(d.dose_rank, d[y])[0]))
        DM = pd.DataFrame(dm); DM['q_slope'] = bh(DM.p_slope.values) if len(DM) else []
        pd.concat([DT, DM], ignore_index=True).to_csv(f'patch_dose_models{SUF}.csv', index=False)

    PTALL = pd.read_csv(f'patch_table{SUF}.csv')
    run_lib('figstyle.py'); run_lib('fig_patch_dynamics.py'); run_lib('fig_patch_acidity.py')
print(f'step 15 done in {time.time()-T0:.0f}s')
