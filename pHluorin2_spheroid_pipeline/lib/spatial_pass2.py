# Pass 2: spatial features of flat-field-corrected 488/405 ratio maps (HIGHER = MORE ACIDIC).
# Works on 2x2-binned images: 1 binned px = 2 full-res px. Distances reported in FULL-RES px.
import numpy as np, scipy.ndimage as ndi
from scipy import stats
from skimage import measure

def build_field(bgmaps, tp):
    st = np.stack([v for (t, w), v in bgmaps.items() if t == tp])
    f = np.nanmedian(st, axis=0)
    nanm = np.isnan(f)
    if nanm.any():
        idx = ndi.distance_transform_edt(nanm, return_distances=False, return_indices=True)
        f = f[tuple(idx)]
    f = ndi.gaussian_filter(f, 3)
    f = ndi.zoom(f, 4, order=1)
    yy, xx = np.mgrid[:980, :980]
    centre = (yy - 490)**2 + (xx - 490)**2 < 300**2
    return (f / np.median(f[centre])).astype(np.float32), st

EDGE = 5      # binned px excluded at the mask edge (= 10 full-res px)
RIMW = 30     # rim band width, binned px (= 60 full-res px)
ABS_BINS = np.arange(0, 160, 10)          # binned px from edge -> 0..300 full px
NORM_BINS = np.linspace(0, 1, 11)

def moran_I(grid, valid):
    z = np.where(valid, grid - grid[valid].mean(), 0.0)
    num = 0.0; W = 0
    for dy, dx in [(0, 1), (1, 0)]:
        a = z[:z.shape[0]-dy, :z.shape[1]-dx]; b = z[dy:, dx:]
        v = valid[:z.shape[0]-dy, :z.shape[1]-dx] & valid[dy:, dx:]
        num += 2 * (a * b)[v].sum(); W += 2 * v.sum()
    n = valid.sum()
    return float((n / W) * num / (z[valid]**2).sum()) if W and n > 10 else np.nan

def corr_length(res, valid):
    z = np.where(valid, res - res[valid].mean(), 0.0); mk = valid.astype(float)
    F = np.fft.rfft2(z, s=(1024, 1024)); M = np.fft.rfft2(mk, s=(1024, 1024))
    ac = np.fft.irfft2(F * np.conj(F)); nm = np.fft.irfft2(M * np.conj(M))
    ac = np.fft.fftshift(ac / np.maximum(nm, 1)); ac /= ac[512, 512]
    yy, xx = np.indices(ac.shape); rr = np.hypot(yy - 512, xx - 512).astype(int)
    prof = np.bincount(rr.ravel(), ac.ravel()) / np.bincount(rr.ravel())
    below = np.where(prof[:200] < 1/np.e)[0]
    return float(below[0] * 2) if len(below) else np.nan     # full-res px

def well_features(g, r, m, field):
    gs, rs = ndi.gaussian_filter(g, 1), ndi.gaussian_filter(r, 1)
    R = gs / np.maximum(rs, 1) / field
    d = ndi.distance_transform_edt(m); dmax = d.max()
    me = d > EDGE
    f = {}
    # ---- morphology
    p = measure.regionprops(m.astype(np.uint8))[0]
    f['area_px'] = p.area * 4; f['eq_diam_px'] = p.equivalent_diameter_area * 2
    f['circularity'] = 4*np.pi*p.area / p.perimeter**2; f['solidity'] = p.solidity
    f['eccentricity'] = p.eccentricity
    cont = max(measure.find_contours(m.astype(float), 0.5), key=len)
    rad = np.hypot(cont[:, 0] - p.centroid[0], cont[:, 1] - p.centroid[1])
    f['boundary_radius_cv'] = rad.std() / rad.mean()
    # ---- distribution
    v = R[me]
    f.update(ratio_mean=v.mean(), ratio_median=np.median(v), ratio_sd=v.std(), ratio_cv=v.std()/v.mean(),
             ratio_p10=np.percentile(v, 10), ratio_p90=np.percentile(v, 90),
             ratio_iqr=np.subtract(*np.percentile(v, [75, 25])), ratio_skew=stats.skew(v), ratio_kurt=stats.kurtosis(v))
    # ---- radial
    core = d > 0.6 * dmax
    for E, tag in [(0, '_e0'), (EDGE, ''), (15, '_e30')]:
        band = (d > E) & (d <= E + RIMW)
        f['rim_ratio' + tag] = np.median(R[band]); f['rim_minus_core' + tag] = np.median(R[band]) - np.median(R[core])
    f['core_ratio'] = np.median(R[core])
    f['rim_over_core_488'] = np.median(gs[(d > EDGE) & (d <= EDGE+RIMW)]) / np.median(gs[core])
    f['rim_over_core_405'] = np.median(rs[(d > EDGE) & (d <= EDGE+RIMW)]) / np.median(rs[core])
    nd = d / dmax
    f['radial_slope'] = np.polyfit(nd[me], R[me], 1)[0]          # ratio per unit normalised depth (neg = acidic rim)
    abs_prof = np.array([np.median(R[(d > a) & (d <= a + 10)]) if ((d > a) & (d <= a + 10)).sum() > 50 else np.nan
                         for a in ABS_BINS])
    norm_prof = np.array([np.median(R[(nd > a) & (nd <= b)]) for a, b in zip(NORM_BINS[:-1], NORM_BINS[1:])])
    # half-max depth of the rim->core transition (full-res px)
    if np.isfinite(abs_prof[1]):
        half = (abs_prof[1] + f['core_ratio']) / 2
        sgn = np.sign(abs_prof[1] - f['core_ratio'])
        cross = np.where(sgn * (abs_prof[1:] - half) < 0)[0]
        f['transition_depth_px'] = float(ABS_BINS[1 + cross[0]] * 2 + 10) if len(cross) else np.nan
    # ---- heterogeneity after removing the radial trend
    fine = np.linspace(0, 1, 21); cen = (fine[:-1] + fine[1:]) / 2
    tr = np.array([np.median(R[(nd > a) & (nd <= b)]) for a, b in zip(fine[:-1], fine[1:])])
    res = R - np.interp(nd, cen, tr)
    f['resid_sd'] = res[me].std()
    B = 6
    H, W = (980 // B) * B, (980 // B) * B
    blk = res[:H, :W].reshape(H//B, B, W//B, B).mean((1, 3))
    bval = me[:H, :W].reshape(H//B, B, W//B, B).all((1, 3))
    f['moran_I_12px'] = moran_I(blk, bval)
    f['corr_length_px'] = corr_length(res, me)
    rsm = ndi.gaussian_filter(np.where(me, res, 0), 3) / np.maximum(ndi.gaussian_filter(me.astype(float), 3), 1e-3)
    for sgn, tag in [(1, 'acidic'), (-1, 'alkaline')]:
        L = measure.label(me & (sgn * rsm > 0.06))
        areas = np.array([q.area for q in measure.regionprops(L) if q.area >= 40])
        f[f'{tag}_patch_n'] = len(areas)
        f[f'{tag}_patch_frac'] = areas.sum() / me.sum()
        f[f'{tag}_patch_mean_area_px'] = areas.mean() * 4 if len(areas) else 0.0
    # ---- angular (rim band)
    yy, xx = np.indices(m.shape)
    ang = np.arctan2(yy - p.centroid[0], xx - p.centroid[1])
    band = (d > EDGE) & (d <= EDGE + RIMW)
    sec = ((ang[band] + np.pi) / (2*np.pi) * 16).astype(int) % 16
    sm = np.array([np.median(R[band][sec == k]) for k in range(16)])
    f['rim_angular_cv'] = sm.std() / sm.mean()
    f['rim_polarity'] = np.abs(np.fft.fft(sm)[1]) / 16 * 2 / sm.mean()
    w_ = np.clip(R - R[me].min(), 0, None) * me
    cy, cx = (w_ * yy).sum() / w_.sum(), (w_ * xx).sum() / w_.sum()
    f['acid_centroid_offset'] = np.hypot(cy - p.centroid[0], cx - p.centroid[1]) / (p.equivalent_diameter_area / 2)
    return f, abs_prof, norm_prof, R

def pass2_well(tp, w, zi_w, ri_w, m, keep):
    """One well of the legacy pass 2 (worker entry point, lib/parallel.py); plate fields come from PAR_FIELDS."""
    g, r, b = load_well(tp, w, {w: zi_w}, {w: ri_w} if ri_w is not None else {})
    f, ap, npf, R = well_features(g, r, m, PAR_FIELDS[tp])
    return f, ap, npf, (np.where(m, R, np.nan).astype(np.float32) if keep else None)

def run_pass2(p1, masks, fields, keep_maps=()):
    from parallel import WellPool
    rows, absP, normP, maps = [], {}, {}, {}
    tasks = []
    for tp, plate in PLATES.items():
        zi = zproj_index(plate); ri = None
        for w in WELLS:
            if (tp, w) not in masks: continue
            if not all(k in zi.get(w, {}) for k in 'grb') and ri is None:
                ri = raw_index(plate)
            tasks.append((tp, w, zi.get(w, {}), ri.get(w) if ri is not None else None, masks[(tp, w)], (tp, w) in keep_maps))
    with WellPool(HERE, ['common.py', 'spatial_pass2.py'],
                  shared={'PAR_FIELDS': fields}, fallback_ns=globals()) as pool:
        for i, (t, (f, ap, npf, Rm)) in enumerate(zip(tasks, pool.imap('pass2_well', tasks))):
            tp, w = t[0], t[1]
            f.update(timepoint=tp, well=w, condition=COND[w]); rows.append(f)
            absP[(tp, w)], normP[(tp, w)] = ap, npf
            if Rm is not None: maps[(tp, w)] = Rm
            if i + 1 == len(tasks) or tasks[i + 1][0] != tp: print(tp, 'done', flush=True)
    return pd.DataFrame(rows), absP, normP, maps
