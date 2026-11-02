# Compare condition effects (vs same-plate DMSO) between max projection and chromatic-matched equatorial plane.
import numpy as np, pandas as pd
from scipy import stats
ZFEAT = ['ratio_median', 'rim_minus_core', 'resid_sd_rob', 'acid_patch_frac', 'corr_len_rob', 'patch_boundary_enrich']
ZNICE = {'ratio_median': 'mean acidity (median 488/405)', 'rim_minus_core': 'rim − centre 488/405',
         'resid_sd_rob': 'residual SD (detrended)', 'acid_patch_frac': 'acidic-patch area fraction',
         'corr_len_rob': 'correlation length (px)', 'patch_boundary_enrich': 'patch boundary enrichment'}

def effects(df, prefixes=('mp_', 'eq_', 'equ_')):
    out = []
    for tp, d in df.groupby('timepoint'):
        dm = d[d.condition == 'DMSO']
        for c, x in d.groupby('condition'):
            if c == 'DMSO': continue
            r = dict(timepoint=tp, condition=c, n=len(x))
            for f in ZFEAT:
                for p in prefixes:
                    sd = d.groupby('condition')[p + f].std().pow(2).mean() ** 0.5
                    r[p + f] = x[p + f].mean() - dm[p + f].mean(); r[p + f + '_es'] = r[p + f] / sd
            out.append(r)
    return pd.DataFrame(out)

def agreement(E, a='mp_', b='eq_'):
    rows = []
    for f in ZFEAT:
        x, y = E[a + f + '_es'], E[b + f + '_es']; ok = x.notna() & y.notna()
        big = ok & (x.abs() > 1)
        rows.append(dict(feature=f, spearman=stats.spearmanr(x[ok], y[ok])[0], n=int(ok.sum()),
                         sign_agree_all=float((np.sign(x[ok]) == np.sign(y[ok])).mean()),
                         sign_agree_large=float((np.sign(x[big]) == np.sign(y[big])).mean()) if big.any() else np.nan,
                         n_large=int(big.sum()),
                         slope=float(np.polyfit(x[ok], y[ok], 1)[0])))
    return pd.DataFrame(rows)
