# Follow individual spheroids (same well) across 24/48/72 h: is the spatial patch pattern persistent?
import numpy as np, pandas as pd, itertools
from scipy import ndimage as ndi

NG = 48
_u = np.linspace(-1, 1, NG); UU, VV = np.meshgrid(_u, _u, indexing='ij'); DISK = UU**2 + VV**2 <= 0.92**2

def _disk_src(a):
    return np.nan_to_num(a.astype(np.float32), nan=np.nan)

def norm_disk(a, centroid, eq_r, rot_deg=0.0, mirror=False, src=None):
    """src: _disk_src(a), precomputed when the same map is resampled many times (rotation search)."""
    th = np.deg2rad(rot_deg); u, v = UU, (-VV if mirror else VV)
    ur = u*np.cos(th) - v*np.sin(th); vr = u*np.sin(th) + v*np.cos(th)
    yy = centroid[0] + ur*eq_r; xx = centroid[1] + vr*eq_r
    out = ndi.map_coordinates(_disk_src(a) if src is None else src, [yy, xx], order=1, cval=np.nan)
    return np.where(DISK, out, np.nan)

def _r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 else np.nan

def pair_corr(mpA, mpB, rot_search=True):
    A = norm_disk(mpA['res'], mpA['centroid'], mpA['eq_r'])
    srcB = _disk_src(mpB['res'])
    r0 = _r(A, norm_disk(mpB['res'], mpB['centroid'], mpB['eq_r'], src=srcB))
    if not rot_search: return r0, np.nan, np.nan
    best, bang = -2, 0
    for mir in (False, True):
        for ang in range(0, 360, 15):
            rr = _r(A, norm_disk(mpB['res'], mpB['centroid'], mpB['eq_r'], ang, mir, src=srcB))
            if np.isfinite(rr) and rr > best: best, bang = rr, (ang if not mir else -ang)
    return r0, best, bang

def patch_polar(mp):
    """residual-weighted centroid of positive (acidic) residual, in normalised polar coords."""
    A = norm_disk(mp['res'], mp['centroid'], mp['eq_r'])
    w = np.clip(np.nan_to_num(A), 0, None)
    if w.sum() == 0: return np.nan, np.nan, np.nan
    cy, cx = (w*UU).sum()/w.sum(), (w*VV).sum()/w.sum()
    rn = np.sqrt(UU**2 + VV**2)
    return float(np.hypot(cy, cx)), float(np.degrees(np.arctan2(cy, cx))), float((w*rn).sum()/w.sum())

def tracking_tables(MAPS, keys, tag, cond_of, pairs=None, null_per_pair=400, seed=0):
    rng = np.random.default_rng(seed); same, null = [], []
    if pairs is None:   # consecutive days, then first -> last (original: 24-48, 48-72, 24-72)
        pairs = list(zip(TPS[:-1], TPS[1:])) + ([(TPS[0], TPS[-1])] if len(TPS) > 2 else [])
    for t1, t2 in pairs:
        w1 = sorted({w for (t, w) in keys if t == t1}); w2 = sorted({w for (t, w) in keys if t == t2})
        both = [w for w in w1 if w in w2]
        for w in both:
            r0, rb, ang = pair_corr(MAPS[(t1, w)][tag], MAPS[(t2, w)][tag])
            c1 = MAPS[(t1, w)][tag]['centroid']; c2 = MAPS[(t2, w)][tag]['centroid']
            same.append(dict(pair=f'{t1}->{t2}', well=w, condition=cond_of[w], r_fixed=r0, r_rotmax=rb, best_rot=ang,
                             centroid_shift_px=4*float(np.hypot(c1[0]-c2[0], c1[1]-c2[1]))))
        allp = [(a, b) for a in w1 for b in w2 if a != b]
        idx = rng.choice(len(allp), size=min(null_per_pair, len(allp)), replace=False)
        for ii in idx:
            a, b = allp[ii]
            r0, rb, _ = pair_corr(MAPS[(t1, a)][tag], MAPS[(t2, b)][tag])
            null.append(dict(pair=f'{t1}->{t2}', well_a=a, well_b=b, same_condition=cond_of[a] == cond_of[b],
                             r_fixed=r0, r_rotmax=rb))
    return pd.DataFrame(same), pd.DataFrame(null)
