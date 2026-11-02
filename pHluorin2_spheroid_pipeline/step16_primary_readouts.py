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
"""Step 16 - the eight pre-specified primary readouts (README section 'Primary readouts'), in one table with one test
family. Needs step 4 (spatial_features_v2.csv) and step 15 (patch_*.csv).

  1 ratio_median_cal          mean acidity (log2 fold-change vs same-plate DMSO)       primary timepoint
  2 acid_patch_frac_cal       patch area fraction                                     primary timepoint
  3 patch_contrast_depth_mean patch acidity vs same-depth tissue (ln)                 primary timepoint
  4 patch_density_per_mm2     patches per mm2                                         primary timepoint
  5 emerged_per_day           new patches per day                                     primary transition
  6 vanished_per_day          lost patches per day                                    primary transition
  7 disp_per_day              movement of persisting patches (radii per day)          primary transition
  8 rel_growth_patch_per_day  patch growth relative to the spheroid (log2 per day)    primary transition
  9 growth_log2_per_day       spheroid growth (log2 area change per day)              primary transition
 10 signal_log2_per_day       total fluorescence (log2 change per day of the integrated medium-subtracted
                              488 + 405 signal: a cell-mass / viability proxy)        primary transition

Primary timepoint: SPHEROID_PRIMARY_TP (default: last read). Primary transition: SPHEROID_PRIMARY_PAIR (default: last
consecutive pair). Registration for 5-8: SPHEROID_PRIMARY_REGISTER (default 'fixed'; use 'rotreg' if the rotation check
in patch_dynamics_summary.svg shows that spheroids turn between reads).
Test: per compound, OLS readout ~ dose_rank on DMSO (rank 0) + that compound's spheroids (spheroid = unit, HC3 SE);
Benjamini-Hochberg across ALL readouts x compounds (one family). Sensitivity: (a) + log spheroid area as covariate;
(b) readouts 2-8 recomputed without patches flagged signal_loss; (c) + the spheroid's growth over the primary
transition as covariate (does the effect survive accounting for growth inhibition?); (d) readouts 1-4 + the same
spheroid's value at the baseline read (SPHEROID_BASELINE_TP, default first read) as covariate, which removes
spheroid-to-spheroid variation. (d) is valid only if the drug has not yet acted at the baseline read: the dose trend AT
the baseline read is reported beside it (slope_baseline_effect, p_baseline_effect) - if that is significant, use (d)
with caution. Spheroids with qc_exclude and Fluorobrite are left out.
Early predicts late (early_predicts_late.csv): within condition (condition as a fixed effect, spheroid = unit), do
features at an early read (SPHEROID_EARLY_TP; default the second read if the first is < 12 h, else the first)
predict LATER changes in growth, total signal, mean ratio and patch fraction? The outcome window starts at the read
AFTER the early read (e.g. 24 h features -> 48->72 h change), so the early measurement is not part of the outcome:
using the same read in both creates regression-to-the-mean / shared-error correlations (on synthetic plates with no
designed link this produced rho up to 0.86). Early spheroid size (log area) is a covariate in every model.
Statistic: Spearman correlation of the predictor and outcome after removing condition means and early size (partial,
within condition). p: permutation test, the predictor shuffled among spheroids of the SAME condition (2000 times) -
exact for 'no within-condition association' however skewed or zero-inflated the feature (model-based p-values gave
false positives on synthetic plates with 2-3 spheroids per condition). Observational; BH across all pairs.
Secondary readouts (secondary_readouts_tests.csv, its own BH family): alkaline patches (step 15 *_alk), bright-field
texture and surface puncta (step 17), if those outputs exist.
Outputs: primary_readouts_per_spheroid.csv, primary_readouts_tests.csv, primary_readouts.svg, early_predicts_late.csv,
early_predicts_late.svg, secondary_readouts_tests.csv"""
run_lib('common.py'); run_lib('stage2_features.py'); run_lib('analysis_stats.py')
PTP = os.environ.get('SPHEROID_PRIMARY_TP', LAST)
PPAIR = os.environ.get('SPHEROID_PRIMARY_PAIR', LASTPAIR)
PREG = os.environ.get('SPHEROID_PRIMARY_REGISTER', 'fixed')
Fv = pd.read_csv('spatial_features_v2.csv'); SP = pd.read_csv('patch_per_spheroid.csv'); PT = pd.read_csv('patch_table.csv')
TR = pd.read_csv('patch_transitions.csv'); LK = pd.read_csv('patch_links.csv')
READOUTS = [('ratio_median_cal', 'tp', 'mean acidity (log2 FC vs DMSO)'), ('acid_patch_frac_cal', 'tp', 'patch area fraction'),
            ('patch_contrast_depth_mean', 'tp', 'patch contrast vs same-depth tissue (ln)'),
            ('patch_density_per_mm2', 'tp', 'patches per mm²'), ('emerged_per_day', 'tr', 'new patches per day'),
            ('vanished_per_day', 'tr', 'lost patches per day'), ('disp_per_day', 'tr', 'patch movement (radii per day)'),
            ('rel_growth_patch_per_day', 'tr', 'patch growth vs spheroid (log2 per day)'),
            ('growth_log2_per_day', 'tr', 'spheroid growth (log2 area per day)'),
            ('signal_log2_per_day', 'tr', 'total fluorescence (log2 per day)')]
BASE_TP = os.environ.get('SPHEROID_BASELINE_TP', TPS[0])
EARLY_TP = os.environ.get('SPHEROID_EARLY_TP', TPS[1] if HOURS[TPS[0]] < 12 and len(TPS) > 2 else TPS[0])
# spheroid-level per read: area and integrated medium-subtracted fluorescence (488 + 405)
W = Fv[['timepoint', 'well', 'area_px', 'g_mean', 'r_mean', 'g_medium', 'r_medium', 'ratio_median_cal', 'acid_patch_frac_cal', 'qc_exclude']].copy()
W['total_signal'] = ((W.g_mean - W.g_medium) + (W.r_mean - W.r_medium)) * W.area_px
WI = W.set_index(['timepoint', 'well'])
def change_per_day(col, ta, tb, log=True):
    a, b = WI.xs(ta)[col], WI.xs(tb)[col]; ok = ~WI.xs(ta).qc_exclude.astype(bool) & ~WI.xs(tb).qc_exclude.astype(bool)
    days = (HOURS[tb] - HOURS[ta]) / 24
    v = (np.log2(b / a) if log else (b - a)) / days
    return v.where(ok & ((a > 0) & (b > 0) if log else True))
print(f'primary timepoint {PTP}, primary transition {PPAIR}, registration {PREG}', flush=True)

# ---- per spheroid, primary timepoint
key = ['timepoint', 'well']
A = Fv[(Fv.timepoint == PTP)][key + ['condition', 'compound', 'dose_rank', 'qc_exclude', 'area_px', 'ratio_median_cal',
                                     'acid_patch_frac_cal']].merge(
    SP[SP.timepoint == PTP][key + ['patch_contrast_depth_mean', 'patch_density_per_mm2', 'valid_area_um2', 'frac_signal_loss']], on=key, how='left')
dm = A[(A.condition == 'DMSO') & ~A.qc_exclude].ratio_median_cal.mean()
A['ratio_median_cal'] = np.log2(A.ratio_median_cal / dm)                    # readout 1 as log2 fold-change vs DMSO
# sensitivity (b): readouts 2-4 without signal_loss patches
pk = PT[(PT.timepoint == PTP)]; keep = pk[~pk.signal_loss.astype(bool)]
g = keep.groupby('well'); s2 = pd.DataFrame({'patch_area_keep': g.area_um2.sum(), 'n_keep': g.size(), 'contrast_keep': g.log_contrast_depth.mean()})
A = A.merge(s2, left_on='well', right_index=True, how='left').fillna({'patch_area_keep': 0, 'n_keep': 0})
A['acid_patch_frac_cal__nosl'] = A.patch_area_keep / A.valid_area_um2      # step-15 patch units (same pixels as step 4)
A['patch_contrast_depth_mean__nosl'] = A.contrast_keep
A['patch_density_per_mm2__nosl'] = A.n_keep / A.valid_area_um2 * 1e6
# ---- per spheroid, primary transition
B = TR[(TR.pair == PPAIR) & (TR.register == PREG)][['well', 'emerged_per_day', 'vanished_per_day', 'disp_per_day',
                                                   'rel_growth_patch_per_day', 'dt_h']]
t1, t2 = PPAIR.split('->')
flag = {(r.timepoint, r.well, r.label): bool(r.signal_loss) for r in PT.itertuples()}
L = LK[(LK.pair == PPAIR) & (LK.register == PREG)].copy()
L['flagA'] = [flag.get((t1, w, a), False) if a > 0 else False for w, a in zip(L.well, L.labelA)]
L['flagB'] = [flag.get((t2, w, b), False) if b > 0 else False for w, b in zip(L.well, L.labelB)]
Lk = L[~L.flagA & ~L.flagB]; days = (HOURS[t2] - HOURS[t1]) / 24
gl = Lk.groupby('well')
s3 = pd.DataFrame({'emerged_per_day__nosl': gl.event.apply(lambda e: (e == 'emerged').sum()) / days,
                   'vanished_per_day__nosl': gl.event.apply(lambda e: (e == 'vanished').sum()) / days,
                   'disp_per_day__nosl': Lk[Lk.event == 'persist'].groupby('well').disp.median() / days,
                   'rel_growth_patch_per_day__nosl': Lk[Lk.event == 'persist'].groupby('well').rel_log2fc.mean() / days})
B = B.merge(s3, left_on='well', right_index=True, how='left')
B = B.merge(pd.DataFrame({'growth_log2_per_day': change_per_day('area_px', t1, t2),
                          'signal_log2_per_day': change_per_day('total_signal', t1, t2)}), left_on='well', right_index=True, how='outer')
# transitions need both reads QC-passed (step 15 already requires it); join spheroid info from the primary timepoint
D = A.merge(B, on='well', how='left')
D['log_area'] = np.log(D.area_px)
# (d) baseline values of readouts 1-4 (same spheroid, baseline read)
if BASE_TP != PTP:
    Bs = Fv[Fv.timepoint == BASE_TP][['well', 'condition', 'qc_exclude', 'ratio_median_cal', 'acid_patch_frac_cal']].merge(
        SP[SP.timepoint == BASE_TP][['well', 'patch_contrast_depth_mean', 'patch_density_per_mm2']], on='well', how='left')
    dmb = Bs[(Bs.condition == 'DMSO') & ~Bs.qc_exclude].ratio_median_cal.mean()
    Bs['ratio_median_cal'] = np.log2(Bs.ratio_median_cal / dmb)
    Bs.loc[Bs.qc_exclude.astype(bool), ['ratio_median_cal', 'acid_patch_frac_cal', 'patch_contrast_depth_mean', 'patch_density_per_mm2']] = np.nan
    D = D.merge(Bs.drop(columns=['condition', 'qc_exclude']).rename(columns=lambda c: c if c == 'well' else c + '__base'), on='well', how='left')
D.to_csv('primary_readouts_per_spheroid.csv', index=False)

# ---- tests: one family
Dq = D[~D.qc_exclude & (D.condition != 'Fluorobrite')]
rows = []
for y, lvl, lab in READOUTS:
    for c in COMPOUNDS:
        d = Dq[Dq.compound.isin(['DMSO', c])]
        r = dict(readout=y, label=lab, level=('timepoint ' + PTP) if lvl == 'tp' else f'transition {PPAIR} ({PREG})', compound=c)
        for tag, form, col in [('', f'{y} ~ dose_rank', y), ('_adj_area', f'{y} ~ dose_rank + log_area', y),
                               ('_no_signal_loss', f'{y}__nosl ~ dose_rank', y + '__nosl'),
                               ('_adj_growth', f'{y} ~ dose_rank + growth_log2_per_day', y),
                               ('_adj_baseline', f'{y} ~ dose_rank + {y}__base', y),
                               ('_baseline_effect', f'{y}__base ~ dose_rank', y + '__base')]:
            if col not in d: continue
            if tag == '_adj_growth' and y == 'growth_log2_per_day': continue
            if tag in ('_adj_baseline', '_baseline_effect') and y + '__base' not in d: continue
            dd = d.dropna(subset=[col] + ([y + '__base'] if tag == '_adj_baseline' else []) + (['growth_log2_per_day'] if tag == '_adj_growth' else []))
            if dd.compound.nunique() < 2 or dd[col].nunique() < 2: continue
            f = smf.ols(form, dd).fit(cov_type='HC3')
            r.update({f'n{tag}': len(dd), f'slope{tag}': f.params['dose_rank'], f'se{tag}': f.bse['dose_rank'], f'p{tag}': f.pvalues['dose_rank']})
            if tag == '':
                r['dmso_mean'] = dd[dd.compound == 'DMSO'][col].mean()
                r['top_dose_mean'] = dd[dd.dose_rank == dd.dose_rank.max()][col].mean()
                r['rho'] = stats.spearmanr(dd.dose_rank, dd[col])[0]
        rows.append(r)
TT = pd.DataFrame(rows)
TT['q'] = bh(TT.p.values)                                     # one family: all readouts x compounds
for tag in ['_adj_area', '_no_signal_loss', '_adj_growth', '_adj_baseline', '_baseline_effect']:
    if 'p' + tag in TT: TT['q' + tag] = bh(TT['p' + tag].values)
TT.to_csv('primary_readouts_tests.csv', index=False)
pd.set_option('display.width', 220)
show = TT[TT.q < 0.05][[c for c in ['readout', 'compound', 'dmso_mean', 'top_dose_mean', 'slope', 'q', 'q_adj_area', 'q_no_signal_loss',
                         'q_adj_growth', 'q_adj_baseline', 'q_baseline_effect'] if c in TT]]
print(f'{len(show)} of {len(TT)} readout x compound tests with q < 0.05:'); print(show.round(4).to_string(index=False))

# ---- figure: one panel per readout, condition means at the primary timepoint / transition
run_lib('figstyle.py'); apply_figure_style(sizes=(7, 6, 5.5))
order = [c for c in display_order if c in set(Dq.condition)]; xo = {c: i for i, c in enumerate(order)}
INFO = {c: cond_info(c) for c in order}; pal = plt.get_cmap('tab20')
fig, axes = plt.subplots(5, 2, figsize=(15, 19)); fig.subplots_adjust(hspace=0.9, wspace=0.12, left=0.06, right=0.98, top=0.96, bottom=0.05)
for ax, (y, lvl, lab) in zip(axes.ravel(), READOUTS):
    for c, g in Dq.groupby('condition'):
        if c not in xo: continue
        col = '0.25' if INFO[c][0] == 'DMSO' else pal(COMPOUNDS.index(INFO[c][0]) % 20 if INFO[c][0] in COMPOUNDS else 0)
        v = g[y].dropna(); xs = xo[c] + (np.linspace(-0.2, 0.2, len(v)) if len(v) > 1 else np.zeros(len(v)))
        ax.scatter(xs, v, s=7, color=col, lw=0); ax.plot([xo[c] - 0.3, xo[c] + 0.3], [v.mean()] * 2, color='k', lw=1)
    ref = Dq[Dq.condition == 'DMSO'][y].mean()
    if np.isfinite(ref): ax.axhline(ref, color='0.6', ls=':', lw=0.7)
    for c in COMPOUNDS:                                      # mark compounds whose dose trend survives the family-wide BH
        t = TT[(TT.readout == y) & (TT.compound == c)]
        if len(t) and t.q.iloc[0] < 0.05:
            xs = [xo[k] for k in order if INFO[k][0] == c]
            ax.annotate('*', (np.mean(xs), 1.0), xycoords=('data', 'axes fraction'), ha='center', fontsize=11, color='tab:red')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
    ax.set_title(f"{lab}  [{('at ' + PTP) if lvl == 'tp' else PPAIR}]", loc='left', fontsize=7)
fig.suptitle(f'Primary readouts (spheroid = point, bar = condition mean, dotted = DMSO; * = dose trend q < 0.05, BH over all '
             f'{len(TT)} readout x compound tests)', fontsize=8)
fig.savefig('primary_readouts.svg', bbox_inches='tight'); plt.close(fig)

# ---- early predicts late (within condition)
print(f'== early ({EARLY_TP}) features vs later change', flush=True)
E = Fv[Fv.timepoint == EARLY_TP][['well', 'condition', 'compound', 'qc_exclude', 'ratio_median_cal', 'acid_patch_frac_cal', 'alk_patch_frac_cal']].merge(
    SP[SP.timepoint == EARLY_TP][['well', 'patch_contrast_depth_mean', 'patch_density_per_mm2', 'frac_signal_loss']], on='well', how='left')
E['ratio_log2_vs_dmso'] = np.log2(E.ratio_median_cal / E[(E.condition == 'DMSO') & ~E.qc_exclude].ratio_median_cal.mean())
PRED = ['ratio_log2_vs_dmso', 'acid_patch_frac_cal', 'alk_patch_frac_cal', 'patch_contrast_depth_mean', 'patch_density_per_mm2', 'frac_signal_loss']
if os.path.exists('texture_features.csv'):
    TXf = pd.read_csv('texture_features.csv')
    tcols = ['bf_core_vs_rim_log2', 'bf_texture_core', 'dark_core_frac', 'puncta_per_1000um2', 'frac_acidic_puncta', 'rim_cellscale_sd']
    E = E.merge(TXf[TXf.timepoint == EARLY_TP][['well'] + tcols], on='well', how='left'); PRED += tcols
OUT_START = TPS[TPS.index(EARLY_TP) + 1] if TPS.index(EARLY_TP) + 1 < len(TPS) else None   # outcome window: next read -> last
OUT = {}
if OUT_START is not None and OUT_START != LAST:
    OUT = {'growth_log2_per_day': change_per_day('area_px', OUT_START, LAST), 'signal_log2_per_day': change_per_day('total_signal', OUT_START, LAST),
           'ratio_log2_per_day': change_per_day('ratio_median_cal', OUT_START, LAST),
           'patch_frac_change_per_day': change_per_day('acid_patch_frac_cal', OUT_START, LAST, log=False)}
    E = E.merge(pd.DataFrame(OUT), left_on='well', right_index=True, how='left')
    E = E.merge(W[W.timepoint == EARLY_TP][['well', 'area_px']].rename(columns={'area_px': 'area_early'}), on='well', how='left')
    E['log_area_early'] = np.log(E.area_early)
else:
    print(f'early predicts late: needs a read between {EARLY_TP} and {LAST} to start the outcome window; skipped')
Eq = E[~E.qc_exclude.astype(bool) & (E.condition != 'Fluorobrite')]
er = []
if OUT:
    for x in PRED:
        for y in OUT:
            d = Eq.dropna(subset=[x, y, 'log_area_early'])
            d = d[d.groupby('condition').well.transform('size') >= 2]       # within-condition contrasts need >= 2 spheroids
            if len(d) < 10 or d[x].nunique() < 3: continue
            f = smf.ols(f'{y} ~ {x} + log_area_early + C(condition)', d).fit()
            res_ = lambda v: smf.ols(f'{v} ~ log_area_early + C(condition)', d).fit().resid.values
            rx, ry = stats.rankdata(res_(x)), stats.rankdata(res_(y))
            rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std(); rho = float(np.mean(rx * ry))
            # permutation within condition (fixed seed per pair for reproducibility)
            prng = np.random.default_rng(PRED.index(x) * 100 + list(OUT).index(y))
            groups = [np.flatnonzero(d.condition.values == c) for c in d.condition.unique()]
            null = np.empty(2000)
            for b in range(2000):
                perm = np.arange(len(rx))
                for gi in groups: perm[gi] = prng.permutation(gi)
                null[b] = np.mean(rx[perm] * ry)
            er.append(dict(early_tp=EARLY_TP, outcome_window=f'{OUT_START}->{LAST}', predictor=x, outcome=y, n_spheroids=len(d),
                           slope_within_condition=f.params[x], rho_within_condition=rho,
                           p=float((np.sum(np.abs(null) >= abs(rho)) + 1) / (len(null) + 1))))
EP = pd.DataFrame(er)
if len(EP):
    EP['q'] = bh(EP.p.values); EP.to_csv('early_predicts_late.csv', index=False)
    print(EP[EP.q < 0.05].round(4).to_string(index=False) if (EP.q < 0.05).any() else 'no early feature predicts a later change at q < 0.05')
    H = EP.pivot(index='predictor', columns='outcome', values='rho_within_condition').reindex(index=[p for p in PRED if p in set(EP.predictor)])
    Qm = EP.pivot(index='predictor', columns='outcome', values='q').reindex(index=H.index)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(H) + 1.6))
    im = ax.imshow(H.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.isfinite(H.values[i, j]):
                ax.text(j, i, f'{H.values[i, j]:.2f}' + ('*' if Qm.values[i, j] < 0.05 else ''), ha='center', va='center', fontsize=6)
    ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(H.columns, rotation=30, ha='right', fontsize=6)
    ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label('within-condition Spearman rho', fontsize=6)
    ax.set_title(f'Early ({EARLY_TP}) features vs later change {OUT_START} -> {LAST}\n(within condition, adjusted for early size; * q < 0.05)', fontsize=7, loc='left')
    fig.savefig('early_predicts_late.svg', bbox_inches='tight'); plt.close(fig)

# ---- secondary readouts (own BH family): alkaline patches, bright-field texture, surface puncta
SEC = []
if os.path.exists('patch_per_spheroid_alk.csv'):
    SA = pd.read_csv('patch_per_spheroid_alk.csv'); TA_ = pd.read_csv('patch_transitions_alk.csv')
    X = SA[SA.timepoint == PTP][['well', 'patch_density_per_mm2', 'patch_frac', 'patch_contrast_depth_mean']].rename(columns=lambda c: c if c == 'well' else 'alk_' + c)
    X = X.merge(TA_[(TA_.pair == PPAIR) & (TA_.register == PREG)][['well', 'emerged_per_day', 'vanished_per_day', 'disp_per_day']].rename(
        columns=lambda c: c if c == 'well' else 'alk_' + c), on='well', how='outer')
    SEC.append(X)
if os.path.exists('texture_features.csv'):
    TXf = pd.read_csv('texture_features.csv')
    SEC.append(TXf[TXf.timepoint == PTP][['well', 'bf_core_vs_rim_log2', 'bf_texture_core', 'bf_glcm_contrast', 'dark_core_frac', 'bf_patch_vs_depth_log2',
                                          'puncta_per_1000um2', 'acidic_puncta_per_1000um2', 'frac_acidic_puncta', 'puncta_contrast_median', 'rim_cellscale_sd']])
if SEC:
    Y = Dq[['well', 'condition', 'compound', 'dose_rank']]
    for X in SEC: Y = Y.merge(X, on='well', how='left')
    rows = []
    for y in [c for c in Y.columns if c not in ('well', 'condition', 'compound', 'dose_rank')]:
        for c in COMPOUNDS:
            d = Y[Y.compound.isin(['DMSO', c])].dropna(subset=[y])
            if d.compound.nunique() < 2 or d[y].nunique() < 2: continue
            f = smf.ols(f'{y} ~ dose_rank', d).fit(cov_type='HC3')
            rows.append(dict(readout=y, compound=c, n=len(d), dmso_mean=d[d.compound == 'DMSO'][y].mean(),
                             top_dose_mean=d[d.dose_rank == d.dose_rank.max()][y].mean(), slope=f.params['dose_rank'], p=f.pvalues['dose_rank']))
    ST = pd.DataFrame(rows); ST['q'] = bh(ST.p.values); ST.to_csv('secondary_readouts_tests.csv', index=False)
    print(f'secondary readouts: {int((ST.q < 0.05).sum())} of {len(ST)} tests with q < 0.05')
    print(ST[ST.q < 0.05].round(4).to_string(index=False))
print(f'step 16 done in {time.time()-T0:.0f}s')
