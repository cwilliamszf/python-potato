"""Step 8 - compare compound effects across experiments (MCF7 exp1, MCF7 repeat, HEK293) and projections.
Readout: 488/405, HIGHER = MORE ACIDIC. Effects are per condition x timepoint, relative to the same-plate DMSO wells.
Inputs : <results dir>/spatial_features_v2.csv of each dataset (steps 3-4).
Outputs (in SPHEROID_COMPARE_DIR): effects_all_datasets.csv, replication_summary.csv, cellline_comparison.csv,
         projection_agreement.csv, dose_trend_by_timepoint.csv, medium_405_vs_dose.csv, compound_summary_table.csv,
         fig_replication_mcf7.svg, fig_cellline_heatmap.svg, fig_timecourse_datasets.svg, fig_ogm_medium_405.svg,
         fig_projection_agreement.svg"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use('Agg')
import matplotlib as mpl, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'lib'))
from figstyle import apply_figure_style, panel_letter
warnings.filterwarnings('ignore')
apply_figure_style(sizes=(7, 6, 5.5)); mpl.rcParams['svg.fonttype'] = 'none'

FU = os.environ.get('SPHEROID_RESULTS_ROOT', os.path.join(HERE, 'results_followup'))
DATASETS = {   # name -> (results dir, cell line, experiment, projection)
    'MCF7_exp1_max': (os.environ.get('SPHEROID_EXP1_DIR', os.path.join(HERE, 'results')), 'MCF7', 'exp1', 'max'),
    'MCF7_exp2_max': (os.path.join(FU, 'MCF7_exp2_repeat_max'), 'MCF7', 'exp2', 'max'),
    'MCF7_exp2_eq': (os.path.join(FU, 'MCF7_exp2_repeat_equatorial'), 'MCF7', 'exp2', 'equatorial'),
    'HEK293_max': (os.path.join(FU, 'HEK293_exp1_max'), 'HEK293', 'exp1', 'max'),
    'HEK293_eq': (os.path.join(FU, 'HEK293_exp1_equatorial'), 'HEK293', 'exp1', 'equatorial')}
SBR_MIN = float(os.environ.get('SPHEROID_SBR_MIN', 1.15))
OUT = os.environ.get('SPHEROID_COMPARE_DIR', os.path.join(FU, 'comparison'))
os.makedirs(OUT, exist_ok=True); os.chdir(OUT)

def savefig_(fig, name):
    fig.savefig(name + '.svg', bbox_inches='tight')
    if os.environ.get('SPHEROID_PREVIEW_PNG'): fig.savefig(name + '_preview.png', dpi=70, bbox_inches='tight')

FEATS = {'ratio_median_cal': 'mean acidity (median 488/405)', 'ratio_ws_raw': 'whole-spheroid 488/405 (raw, no bg)',
         'rim_minus_core_cal': 'rim − core 488/405', 'resid_sd_rob_cal': 'residual SD (detrended)',
         'acid_patch_frac_cal': 'acidic-patch area fraction', 'corr_len_rob_cal': 'correlation length (px)'}
CPD_NAME = {'Oz': 'Ouabain', 'Baf': 'Bafilomycin A1', 'Rot': 'Rotenone', '3BP': '3-Bromopyruvate', 'Phlor': 'Phloretin',
            'Olig': 'Oligomycin', 'Ace': 'Acetazolamide', 'OGM': 'OGM', 'Ogerin': 'Ogerin', 'TMZ': 'TMZ', 'Fluorobrite': 'Fluorobrite'}
CPD_ORDER = ['Fluorobrite', 'TMZ', 'Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace', 'OGM', 'Ogerin']
CC = {'Oz': '#E6550D', 'Baf': '#CB181D', 'Rot': '#6A51A3', '3BP': '#8C510A', 'Phlor': '#C51B7D', 'Olig': '#2171B5',
      'Ace': '#B8860B', 'OGM': '#238B45', 'Ogerin': '#17BECF', 'TMZ': '0.55', 'Fluorobrite': '0.75'}
DOSE_LABEL = {}

def fmt_dose(um):
    if um == 0: return 'vehicle'
    return f'{um*1000:g} nM' if um < 1 else f'{um:g} µM'

# ------------------------------------------------------------------ load
F = {}
for nm, (d, cl, ex, pj) in DATASETS.items():
    p = os.path.join(d, 'spatial_features_v2.csv')
    if not os.path.exists(p):
        print('missing', nm, p); continue
    f = pd.read_csv(p); f['ratio_ws_raw'] = f.g_mean / f.r_mean
    # common reliability rule for all datasets: both channels >= SBR_MIN x medium (exp 1 originally used 1.03)
    f['low_signal_common'] = (f.sbr488 < SBR_MIN) | (f.sbr405 < SBR_MIN)
    f['qc_exclude'] = f.qc_exclude | f.low_signal_common
    f['dataset'] = nm; f['cell_line'] = cl; f['experiment'] = ex; f['projection'] = pj
    F[nm] = f
ALL = pd.concat(F.values(), ignore_index=True)
print({k: (len(v), int(v.qc_exclude.sum()), sorted(v.timepoint.unique())) for k, v in F.items()})

# ------------------------------------------------------------------ effects vs same-plate DMSO
rows = []
for nm, f in F.items():
    g = f[~f.qc_exclude]
    for tp, gt in g.groupby('timepoint'):
        ctrl = gt[gt.condition == 'DMSO']
        for cond, gc_ in gt.groupby('condition'):
            if cond == 'DMSO': continue
            r = dict(dataset=nm, cell_line=F[nm].cell_line.iloc[0], experiment=F[nm].experiment.iloc[0], projection=F[nm].projection.iloc[0],
                     timepoint=tp, condition=cond, compound=gc_.compound.iloc[0], dose_um=gc_.dose_um.iloc[0], dose_rank=gc_.dose_rank.iloc[0],
                     n=len(gc_), n_dmso=len(ctrl))
            for ft in FEATS:
                a, b = gc_[ft].dropna(), ctrl[ft].dropna()
                if len(a) < 2 or len(b) < 2: continue
                sp = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
                r[f'{ft}_delta'] = a.mean() - b.mean()
                r[f'{ft}_fc'] = a.mean() / b.mean() if ft.startswith('ratio') else np.nan
                r[f'{ft}_es'] = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
                r[f'{ft}_p'] = stats.ttest_ind(a, b, equal_var=False).pvalue
            rows.append(r)
E = pd.DataFrame(rows)
for ft in FEATS:
    E[f'{ft}_q'] = np.nan
    for nm in E.dataset.unique():
        ix = E.dataset == nm; p = E.loc[ix, f'{ft}_p'].values; ok = np.isfinite(p)
        q = np.full_like(p, np.nan)
        if ok.sum():
            o = np.argsort(p[ok]); pr = p[ok][o] * ok.sum() / np.arange(1, ok.sum() + 1)
            pr = np.minimum.accumulate(pr[::-1])[::-1]; qq = np.empty_like(pr); qq[o] = np.minimum(pr, 1); q[ok] = qq
        E.loc[ix, f'{ft}_q'] = q
E.to_csv('effects_all_datasets.csv', index=False)

# ------------------------------------------------------------------ pairwise agreement
def agree(A, B, label, tps=None, feats=FEATS):
    out = []
    m = A.merge(B, on=['timepoint', 'condition'], suffixes=('_a', '_b'))
    if tps is not None: m = m[m.timepoint.isin(tps)]
    for ft in feats:
        x, y = m[f'{ft}_es_a'], m[f'{ft}_es_b']; ok = x.notna() & y.notna()
        if ok.sum() < 5: continue
        big = ok & (x.abs() > 1.5)
        out.append(dict(comparison=label, feature=ft, n=int(ok.sum()), spearman=stats.spearmanr(x[ok], y[ok])[0],
                        pearson=np.corrcoef(x[ok], y[ok])[0, 1], sign_agree_all=float((np.sign(x[ok]) == np.sign(y[ok])).mean()),
                        n_large_a=int(big.sum()), sign_agree_large_a=float((np.sign(x[big]) == np.sign(y[big])).mean()) if big.any() else np.nan))
    return out, m

ag = []
dsE = {nm: E[E.dataset == nm] for nm in E.dataset.unique()}
if {'MCF7_exp1_max', 'MCF7_exp2_max'} <= dsE.keys():
    o, M12 = agree(dsE['MCF7_exp1_max'], dsE['MCF7_exp2_max'], 'MCF7 exp1 vs exp2 (max proj)', tps=['24h', '48h', '72h']); ag += o
if {'MCF7_exp1_max', 'MCF7_exp2_eq'} <= dsE.keys():
    o, _ = agree(dsE['MCF7_exp1_max'], dsE['MCF7_exp2_eq'], 'MCF7 exp1 (max) vs exp2 (equatorial plane)', tps=['24h', '48h', '72h']); ag += o
if {'MCF7_exp2_max', 'HEK293_max'} <= dsE.keys():
    o, MCL = agree(dsE['MCF7_exp2_max'], dsE['HEK293_max'], 'MCF7 exp2 vs HEK293 (max proj)'); ag += o
if {'MCF7_exp2_eq', 'HEK293_eq'} <= dsE.keys():
    o, _ = agree(dsE['MCF7_exp2_eq'], dsE['HEK293_eq'], 'MCF7 exp2 vs HEK293 (equatorial plane)'); ag += o
for a, b in [('MCF7_exp2_max', 'MCF7_exp2_eq'), ('HEK293_max', 'HEK293_eq')]:
    if {a, b} <= dsE.keys():
        o, _ = agree(dsE[a], dsE[b], f'{a.split("_")[0]}: max projection vs equatorial plane'); ag += o
AG = pd.DataFrame(ag); AG.to_csv('replication_summary.csv', index=False)
print(AG.round(2).to_string(index=False))

# ------------------------------------------------------------------ per-compound summary (mean acidity + heterogeneity)
def call(es, q):
    if not np.isfinite(es): return 'n.d.'
    if abs(es) >= 1.5 and q < 0.1: return 'acidic' if es > 0 else 'alkaline'
    if abs(es) >= 1.5: return '(acidic)' if es > 0 else '(alkaline)'
    return '–'
summ = []
for nm, e in dsE.items():
    for (cpd, cond), g in e.groupby(['compound', 'condition']):
        rec = dict(dataset=nm, compound=cpd, condition=cond, dose_um=g.dose_um.iloc[0])
        for tp, r in g.set_index('timepoint').iterrows():
            rec[f'mean_{tp}'] = call(r.get('ratio_median_cal_es', np.nan), r.get('ratio_median_cal_q', np.nan))
            rec[f'fc_{tp}'] = r.get('ratio_median_cal_fc', np.nan)
            rec[f'patch_{tp}'] = r.get('acid_patch_frac_cal_delta', np.nan)
        summ.append(rec)
S = pd.DataFrame(summ); S.to_csv('compound_summary_table.csv', index=False)

# ------------------------------------------------------------------ dose trend per timepoint (incl. 2 h), vs on-plate DMSO
# The 2 h plate is an EARLY TREATMENT timepoint, not a baseline: pumps, transporters and enzymes act within minutes, so
# 2 h effects are drug effects (transcriptional change and cell death are unlikely this early). Every timepoint is
# compared with the DMSO wells on its own plate; no spheroid is normalised to its own 2 h value.
DT_FEATS = ['ratio_median_cal', 'rim_minus_core_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal', 'sbr488', 'sbr405', 'log_area']
CPDS_T = ['TMZ', 'Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace', 'OGM', 'Ogerin']
dtr = []
for nm, f in F.items():
    if F[nm].projection.iloc[0] != 'max': continue
    g = f[~f.qc_exclude].copy(); g['log_area'] = np.log(g.area_px)
    for tp, gt in g.groupby('timepoint'):
        for cpd in CPDS_T:
            d = gt[gt.compound.isin(['DMSO', cpd])]
            for ft in DT_FEATS:
                try:
                    m = smf.ols(f'{ft} ~ dose_rank', d).fit(cov_type='HC3'); sl, p = m.params['dose_rank'], m.pvalues['dose_rank']
                except Exception:
                    sl, p = np.nan, np.nan
                sd = d[d.compound == 'DMSO'][ft].std()
                dtr.append(dict(dataset=nm, timepoint=tp, compound=cpd, feature=ft, slope=sl, slope_per_dmso_sd=sl / sd if sd > 0 else np.nan, p=p, n=len(d)))
DTR = pd.DataFrame(dtr); DTR['q'] = np.nan
for _, gg in DTR.groupby(['dataset', 'timepoint', 'feature']):
    p = gg.p.values; ok = np.isfinite(p); q = np.full_like(p, np.nan)
    if ok.sum():
        o = np.argsort(p[ok]); pr = p[ok][o] * ok.sum() / np.arange(1, ok.sum() + 1); pr = np.minimum.accumulate(pr[::-1])[::-1]
        qq = np.empty_like(pr); qq[o] = np.minimum(pr, 1); q[ok] = qq
    DTR.loc[gg.index, 'q'] = q
DTR.to_csv('dose_trend_by_timepoint.csv', index=False)

# ------------------------------------------------------------------ OGM 405-channel medium fluorescence
med = []
for nm, f in F.items():
    d = f.copy(); d['row'] = d.well.str[0]; d['col'] = d.well.str[1:].astype(int)
    for tp, dt in d.groupby('timepoint'):
        ref = dt[~dt.compound.isin(['OGM'])]
        try:
            mg = smf.ols('np.log(g_medium) ~ C(row) + C(col)', ref).fit(); mr = smf.ols('np.log(r_medium) ~ C(row) + C(col)', ref).fit()
        except Exception:
            continue
        dt = dt.assign(g_exp=np.exp(mg.predict(dt)), r_exp=np.exp(mr.predict(dt)))
        for _, r in dt.iterrows():
            med.append(dict(dataset=nm, timepoint=tp, well=r.well, condition=r.condition, compound=r.compound, dose_um=r.dose_um,
                            g_medium_rel=r.g_medium / r.g_exp, r_medium_rel=r.r_medium / r.r_exp))
MED = pd.DataFrame(med); MED.to_csv('medium_405_vs_dose.csv', index=False)

# ================================================================== figures
TPC = {'2h': '#BDBDBD', '24h': '#4477AA', '48h': '#EE6677', '72h': '#228833'}
# --- fig 1: replication MCF7 exp1 vs exp2 (effect sizes)
if 'M12' in globals():
    fts = ['ratio_median_cal', 'rim_minus_core_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal', 'corr_len_rob_cal', 'ratio_ws_raw']
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6)); axes = axes.ravel()
    for ax, ft in zip(axes, fts):
        m = M12
        for tp in ['24h', '48h', '72h']:
            mm = m[m.timepoint == tp]
            ax.scatter(mm[f'{ft}_es_a'], mm[f'{ft}_es_b'], s=12, c=[CC.get(c, '0.5') for c in mm.compound_a], marker={'24h': 'o', '48h': 's', '72h': '^'}[tp],
                       edgecolor='none', alpha=0.85)
        x, y = m[f'{ft}_es_a'], m[f'{ft}_es_b']; ok = x.notna() & y.notna()
        lim = np.nanpercentile(np.abs(np.r_[x[ok], y[ok]]), 99) * 1.1 if ok.any() else 1
        ax.plot([-lim, lim], [-lim, lim], ls='--', color='0.6', lw=0.7); ax.axhline(0, color='0.85', lw=0.6); ax.axvline(0, color='0.85', lw=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        rr = stats.spearmanr(x[ok], y[ok])[0] if ok.sum() > 4 else np.nan
        ax.set_title(f'{FEATS[ft]}\nρ = {rr:.2f}, n = {int(ok.sum())}', fontsize=7)
        ax.set_xlabel('effect vs DMSO, MCF7 exp 1 (pooled-SD units)', fontsize=6); ax.set_ylabel('effect vs DMSO, MCF7 exp 2', fontsize=6)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=CC[c], label=CPD_NAME[c], ms=4) for c in CPD_ORDER if c in set(M12.compound_a)]
    h += [Line2D([], [], marker=mk, ls='', color='0.3', label=tp, ms=4) for tp, mk in [('24 h', 'o'), ('48 h', 's'), ('72 h', '^')]]
    fig.legend(handles=h, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=6)
    for i, ax in enumerate(axes): panel_letter(ax, 'abcdef'[i])
    fig.suptitle('MCF7 replication: compound effects in experiment 1 vs experiment 2 (488/405, higher = more acidic; max projection)', fontsize=8)
    fig.tight_layout(); savefig_(fig, 'fig_replication_mcf7'); plt.close(fig)

# --- fig 2: heatmap of mean-acidity and patch effects, conditions x (dataset, timepoint)
order = ['Fluorobrite', '1uM TMZ'] + [f'{d} {s}' for d in ['Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace'] for s in 'LMH'] + \
        ['0.5uM OGM', '1uM OGM', '5uM OGM', '20uM OGM', '0.5uM Ogerin', '5uM Ogerin', '20uM Ogerin', '50uM Ogerin']
cols = [(nm, tp) for nm in ['MCF7_exp1_max', 'MCF7_exp2_max', 'HEK293_max'] if nm in dsE for tp in ['2h', '24h', '48h', '72h']
        if tp in set(dsE[nm].timepoint)]
fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
for ax, ft, ttl in zip(axes, ['ratio_median_cal', 'acid_patch_frac_cal'], ['mean acidity (488/405)', 'acidic-patch area']):
    Mx = np.full((len(order), len(cols)), np.nan); Q = np.full_like(Mx, np.nan)
    for j, (nm, tp) in enumerate(cols):
        e = dsE[nm][dsE[nm].timepoint == tp].set_index('condition')
        for i, c in enumerate(order):
            if c in e.index: Mx[i, j] = e.loc[c, f'{ft}_es']; Q[i, j] = e.loc[c, f'{ft}_q']
    v = 4
    im = ax.imshow(np.clip(Mx, -v, v), cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    for i in range(len(order)):
        for j in range(len(cols)):
            if np.isfinite(Q[i, j]) and Q[i, j] < 0.1: ax.text(j, i, '•', ha='center', va='center', fontsize=6, color='k')
            if not np.isfinite(Mx[i, j]): ax.text(j, i, '×', ha='center', va='center', fontsize=5, color='0.6')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([tp for _, tp in cols], fontsize=5.5)
    xs = 0
    for nm in ['MCF7_exp1_max', 'MCF7_exp2_max', 'HEK293_max']:
        k = sum(1 for c in cols if c[0] == nm)
        if k:
            ax.text(xs + (k - 1) / 2, -0.9, {'MCF7_exp1_max': 'MCF7 exp 1', 'MCF7_exp2_max': 'MCF7 exp 2', 'HEK293_max': 'HEK293'}[nm], ha='center', va='bottom', fontsize=6.5, fontweight='bold')
            if xs: ax.axvline(xs - 0.5, color='k', lw=1)
            xs += k
    for y in [2, 5, 8, 11, 14, 17, 20, 23, 27]: ax.axhline(y - 0.5, color='white', lw=1)
    ax.set_title(f'{ttl}: effect vs same-plate DMSO\n(pooled-SD units; red = more acidic / more patch; • q < 0.1)', fontsize=7, pad=26)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=5.5)
    cb.set_label('higher = more acidic' if ft == 'ratio_median_cal' else 'higher = more acidic-patch area', fontsize=6)
fig.tight_layout(); savefig_(fig, 'fig_cellline_heatmap'); plt.close(fig)

# --- fig 3: time course of mean acidity FC vs DMSO per compound, highest usable dose, three datasets
fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True); axes = axes.ravel()
dstyle = {'MCF7_exp1_max': ('MCF7 exp 1', '-', 'o'), 'MCF7_exp2_max': ('MCF7 exp 2', '--', 's'), 'HEK293_max': ('HEK293', ':', '^')}
for ax, cpd in zip(axes, ['TMZ', 'Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace', 'OGM', 'Ogerin']):
    for nm, (lab, ls, mk) in dstyle.items():
        if nm not in dsE: continue
        e = dsE[nm][dsE[nm].compound == cpd]
        for k, (dose, g) in enumerate(sorted(e.groupby('dose_um'), key=lambda x: x[0])):
            g = g.assign(h=g.timepoint.str.rstrip('h').astype(float)).sort_values('h')
            col = plt.cm.Greys(0.35 + 0.65 * (k + 1) / max(1, e.dose_um.nunique())) if False else CC[cpd]
            alpha = 0.3 + 0.7 * (k + 1) / max(1, e.dose_um.nunique())
            ax.plot(g.h, g.ratio_median_cal_fc, ls=ls, marker=mk, ms=2.5, lw=0.9, color=CC[cpd], alpha=alpha,
                    label=f'{lab}' if k == e.dose_um.nunique() - 1 else None)
    ax.axhline(1, color='0.6', lw=0.6, ls=':'); ax.set_title(CPD_NAME[cpd], fontsize=7)
    ax.set_xticks([2, 24, 48, 72])
    if cpd in ('TMZ', 'Phlor'): ax.set_ylabel('488/405 fold-change vs DMSO\n(higher = more acidic)', fontsize=6)
    if cpd == 'TMZ': ax.legend(fontsize=5, loc='upper left')
for ax in axes[5:]: ax.set_xlabel('hours (nominal)', fontsize=6)
fig.suptitle('Mean acidity over time — all doses (darker = higher dose); line style = dataset', fontsize=8)
fig.tight_layout(); savefig_(fig, 'fig_timecourse_datasets'); plt.close(fig)

# --- fig 4: OGM / Ogerin medium 405 signal vs dose
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for ax, nm in zip(axes, ['MCF7_exp1_max', 'MCF7_exp2_max', 'HEK293_max']):
    if nm not in F: continue
    m = MED[(MED.dataset == nm) & MED.compound.isin(['OGM', 'Ogerin'])]
    for (cpd, tp), g in m.groupby(['compound', 'timepoint']):
        gg = g.groupby('dose_um').r_medium_rel.mean()
        ax.plot(gg.index, gg.values, marker='o' if cpd == 'OGM' else 's', ms=3, lw=1, color=CC[cpd], alpha={'2h': 1, '24h': 0.85, '48h': 0.6, '72h': 0.4}[tp],
                label=f'{cpd} {tp}')
    ax.set_xscale('log'); ax.axhline(1, color='0.6', lw=0.6, ls=':'); ax.set_title(dstyle[nm][0], fontsize=7)
    ax.set_xlabel('compound (µM)', fontsize=6)
axes[0].set_ylabel('405-channel medium signal\n(× position-matched expectation)', fontsize=6); axes[-1].legend(fontsize=5, ncol=2)
fig.tight_layout(); savefig_(fig, 'fig_ogm_medium_405'); plt.close(fig)

# --- fig 5: max projection vs equatorial plane in the new experiments
pairs = [(a, b) for a, b in [('MCF7_exp2_max', 'MCF7_exp2_eq'), ('HEK293_max', 'HEK293_eq')] if {a, b} <= dsE.keys()]
if pairs:
    fts = ['ratio_median_cal', 'rim_minus_core_cal', 'resid_sd_rob_cal', 'acid_patch_frac_cal']
    fig, axes = plt.subplots(len(pairs), 4, figsize=(10, 2.6 * len(pairs)), squeeze=False)
    for r_, (a, b) in enumerate(pairs):
        m = dsE[a].merge(dsE[b], on=['timepoint', 'condition'], suffixes=('_a', '_b'))
        for c_, ft in enumerate(fts):
            ax = axes[r_, c_]; x, y = m[f'{ft}_es_a'], m[f'{ft}_es_b']; ok = x.notna() & y.notna()
            ax.scatter(x, y, s=8, c=[TPC[t] for t in m.timepoint], edgecolor='none')
            lim = np.nanpercentile(np.abs(np.r_[x[ok], y[ok]]), 99) * 1.1 if ok.any() else 1
            ax.plot([-lim, lim], [-lim, lim], ls='--', color='0.6', lw=0.7); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            rr = stats.spearmanr(x[ok], y[ok])[0] if ok.sum() > 4 else np.nan
            ax.set_title(f'{a.split("_")[0]}: {FEATS[ft]} (ρ = {rr:.2f})', fontsize=6.5)
            ax.set_xlabel('effect, max projection', fontsize=6); ax.set_ylabel('effect, equatorial plane', fontsize=6)
    fig.tight_layout(); savefig_(fig, 'fig_projection_agreement'); plt.close(fig)
# --- onset figure (2 h vs 24-72 h), needs datasets MCF7_exp2_max and HEK293_max
if {'MCF7_exp2_max', 'HEK293_max'} <= set(F):
    DT = DTR.replace({'dataset': {'MCF7_exp2_max': 'MCF7_rep', 'HEK293_max': 'HEK293'}}); CPDS = CPDS_T; os.makedirs('out', exist_ok=True)
    exec(open(os.path.join(HERE, 'lib', 'fig_onset.py'), encoding='utf-8').read(), dict(globals()))
    os.replace('out/onset_2h_vs_late.svg', 'fig_onset_2h_vs_late.svg')
    if os.path.exists('out/onset_preview.png'): os.remove('out/onset_preview.png')
print('step 8 done')
