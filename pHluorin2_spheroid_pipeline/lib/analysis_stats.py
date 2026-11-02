# Statistics on per-spheroid spatial features (calibrated-background model, suffix _cal).
import numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')

COMPOUNDS = globals().get('COMPOUNDS_CFG') or ['Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace', 'OGM', 'Ogerin']
FEAT = {'ratio_median_cal': 'mean acidity (median 488/405)',
        'rim_minus_core_cal': 'rim − centre 488/405',
        'resid_sd_cal': 'residual SD (detrended)',
        'acid_patch_frac_cal': 'acidic-patch area fraction',
        'corr_len_px_cal': 'correlation length (px)',
        'patch_boundary_enrich_cal': 'patch boundary enrichment',
        'area_px': 'spheroid area (px²)'}

def dose_time_model(df, cpd, y):
    d = df[df.compound.isin(['DMSO', cpd])].dropna(subset=[y]).copy()
    d['tpc'] = d.timepoint.str.replace('h', '').astype(int)
    out = dict(compound=cpd, feature=y, n_spheroids=d.well.nunique(), n_obs=len(d))
    try:
        mod = smf.ols(f'{y} ~ C(timepoint) * dose_rank', data=d).fit(cov_type='cluster', cov_kwds={'groups': d.well})
        names = mod.params.index
        ref = sorted(d.timepoint.unique())[0]          # patsy reference level (alphabetical)
        for tp in TPS:
            if tp == ref:
                L = np.array([1.0 if n == 'dose_rank' else 0.0 for n in names])
            else:
                L = np.array([1.0 if n in ('dose_rank', f'C(timepoint)[T.{tp}]:dose_rank') else 0.0 for n in names])
            est = float(L @ mod.params.values); se = float(np.sqrt(L @ mod.cov_params().values @ L))
            out[f'slope_{tp}'] = est; out[f'p_slope_{tp}'] = float(2 * stats.norm.sf(abs(est / se))) if se > 0 else np.nan
        inter = [n for n in names if ':dose_rank' in n]
        R = np.array([[1.0 if n == i else 0.0 for n in names] for i in inter])
        w = mod.wald_test(R, scalar=True); out['p_dose_x_time'] = float(w.pvalue)
        # monotonic dose trend per timepoint (Spearman on spheroids)
        for tp in TPS:
            dd = d[d.timepoint == tp]
            out[f'rho_{tp}'] = float(stats.spearmanr(dd.dose_rank, dd[y])[0]) if dd[y].nunique() > 2 else np.nan
    except Exception as e:
        out['error'] = str(e)[:80]
    return out

def bh(p):
    p = np.asarray(p, float); n = np.isfinite(p).sum(); q = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]; order = idx[np.argsort(p[idx])]
    ranked = p[order] * n / (np.arange(1, n + 1)); ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1); return q

def matched_mean(df, het, meancol='ratio_median_cal', k=5):
    """For every spheroid: compare its heterogeneity with the k spheroids of OTHER compounds at the same
    timepoint whose mean acidity is closest. Returns per-spheroid delta and per-condition summary."""
    rows = []
    for tp, d in df.groupby('timepoint'):
        d = d.dropna(subset=[het, meancol])
        for _, r in d.iterrows():
            pool = d[d.compound != r.compound]
            if r.compound in ('DMSO', 'Fluorobrite'):
                pool = d[~d.compound.isin(['DMSO', 'Fluorobrite'])]
            nn = pool.iloc[np.argsort(np.abs(pool[meancol].values - r[meancol]))[:k]]
            rows.append(dict(timepoint=tp, well=r.well, condition=r.condition, compound=r.compound,
                             mean=r[meancol], mean_gap=float(np.abs(nn[meancol] - r[meancol]).mean()),
                             het=r[het], het_matched=float(nn[het].mean()), delta=float(r[het] - nn[het].mean())))
    P = pd.DataFrame(rows)
    S = P.groupby(['condition', 'timepoint']).agg(n=('delta', 'size'), delta_mean=('delta', 'mean'),
                                                   delta_sd=('delta', 'std'), mean_gap=('mean_gap', 'mean'),
                                                   all_same_sign=('delta', lambda x: bool((x > 0).all() or (x < 0).all()))).reset_index()
    return P, S

def ancova(df, het, meancol='ratio_median_cal'):
    d = df.dropna(subset=[het, meancol, 'area_px']).copy(); d['larea'] = np.log(d.area_px)
    mod = smf.ols(f'{het} ~ C(condition, Treatment("DMSO")) + C(timepoint) + {meancol} + I({meancol}**2) + larea',
                  data=d).fit(cov_type='cluster', cov_kwds={'groups': d.well})
    co = mod.params.filter(like='C(condition'); pv = mod.pvalues.filter(like='C(condition')
    return pd.DataFrame({'condition': [i.split('[T.')[1][:-1] for i in co.index], 'coef_vs_DMSO_adj': co.values,
                         'p': pv.values, 'q_BH': bh(pv.values)}), mod

def icc1(df, y):
    out = {}
    for tp, d in df.groupby('timepoint'):
        d = d.dropna(subset=[y]); g = d.groupby('condition')[y]
        k = g.size().mean(); grand = d[y].mean()
        msb = (g.size() * (g.mean() - grand)**2).sum() / (g.ngroups - 1)
        msw = ((d[y] - g.transform('mean'))**2).sum() / (len(d) - g.ngroups)
        out[tp] = float((msb - msw) / (msb + (k - 1) * msw))
    return out

def loo_classify(df, feats, label='condition'):
    """Leave-one-spheroid-out nearest-centroid classification on per-spheroid trajectories (feats x 3 timepoints)."""
    wide = df.pivot_table(index=['well', label], columns='timepoint', values=feats)
    wide = wide.dropna(); X = wide.values; X = (X - X.mean(0)) / X.std(0); y = wide.index.get_level_values(label).values
    correct = 0; n = 0
    for i in range(len(X)):
        mask = np.ones(len(X), bool); mask[i] = False
        cents = pd.DataFrame(X[mask]).groupby(y[mask]).mean()
        if y[i] not in cents.index: continue
        pred = cents.index[np.argmin(((cents.values - X[i])**2).sum(1))]
        correct += pred == y[i]; n += 1
    chance = 1 / len(np.unique(y))
    return correct / n if n else np.nan, chance, n
