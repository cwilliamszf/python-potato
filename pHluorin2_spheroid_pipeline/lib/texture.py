# Step 17 per-well work: bright-field texture (necrosis / compaction proxies) and full-resolution surface puncta.
#
# Bright field (2x2-binned, 980 x 980 px, 1.39 um/px), normalised to the well's medium level (bf_rel = BF / medium):
#   bf_core_rel, bf_rim_rel      median bf_rel in the core (normalised depth > 0.6) and the rim (<= 0.2)
#   bf_core_vs_rim_log2          log2(core / rim): a necrotic core is darker -> negative
#   bf_texture_core / _rim       SD of the high-pass bf_rel (minus a 4-px Gaussian, inside the mask): granularity
#   bf_glcm_contrast / _homogeneity / _entropy   grey-level co-occurrence texture of the core (32 levels, 2 px, 4 angles)
#   dark_core_frac               fraction of core pixels darker than 0.8 x the rim median
#   bf_patch_vs_depth_log2       bright field inside acid patches vs non-patch tissue at the same depth (log2):
#                                negative = patches sit in darker (denser / necrotic) tissue
# Source image: the Cytation ZProj bright field (a max projection, which lowers the contrast of a dark core) or, when
# the equatorial slice is known (follow-up 'v2' experiments), the bright-field slice at that plane (bf_source column).
#
# Surface puncta (full resolution, 0.70 um/px), in the outer RIM_UM of the projected spheroid, where the line of sight
# runs tangentially through the outer cell layers and the least out-of-focus tissue is stacked on top:
#   signal = image - (floor + alpha x (medium - floor))  (the calibrated background model, per well)
#   detection: Laplacian of Gaussian matched to a ~1 um spot (sigma LOG_SIGMA = 1.3 px); its response to noise is
#   symmetric around zero, so PUNCTA_K x the robust SD (MAD) of the response in the band is a valid threshold; local
#   maxima above it in either channel, detections within 3 px merged. Brightness is measured on a white top-hat
#   (disk radius 4 px = 2.8 um), which removes the smooth signal underneath.
#   per punctum: amp488 / amp405 = top-hat / local signal (relative brightening); contrast_log2 = log2 of the 488/405
#   ratio at the punctum (3x3 px) over the ratio in a 5-9 px ring around it; kind = acidic (> log2 1.15),
#   alkaline (< -log2 1.15) or neutral (brighter without a ratio change).
#   per spheroid: n_puncta, puncta_per_1000um2 (of the band), frac_acidic_puncta, median contrast and amplitudes,
#   rim_cellscale_sd = SD of the band-pass (0.8-4 px) log 488/405 in the band: cell-scale ratio heterogeneity that the
#   4x-binned maps average away (includes camera noise; compare between conditions).
# Optical limit: at ~1 um lateral resolution, puncta smaller than ~1 um appear as ~1 um blurred spots; their
# amplitude is underestimated but they are detected if bright enough.
import numpy as np, scipy.ndimage as ndi, tifffile
from skimage import feature
import fastops as fo

PX_FULL_UM = 1367 / 1960
RIM_UM = 30.0
PUNCTA_K = 5.0
LOG_SIGMA = 1.3
TOPHAT_R = 4
ACID_PUNCTUM = np.log2(1.15)
_DISK = (lambda r: (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r))(TOPHAT_R)


def _maxproj(sl):
    st = None
    for _, fp in sorted(sl):
        im = tifffile.imread(fp); st = im if st is None else np.maximum(st, im, out=st)
    return st


def _slice(sl, z):
    return tifffile.imread(sorted(sl)[z][1])


def load_full(zfiles, raw, zplane, projection, segmentation, bf_channel):
    """Full-resolution 488, 405, bright field + description of the bright-field source."""
    if zfiles is not None and all(k in zfiles for k in 'grb'):
        g, r, b = (tifffile.imread(zfiles[k]) for k in 'grb'); src = 'ZProj'
        return g.astype(np.float32), r.astype(np.float32), b.astype(np.float32), src
    if projection == 'max':
        g, r = _maxproj(raw['GFP']), _maxproj(raw['Oxidized roGFP2'])
    else:
        g, r = _slice(raw['GFP'], zplane), _slice(raw['Oxidized roGFP2'], zplane)
    if zplane is not None and np.isfinite(zplane):
        b = _slice(raw[bf_channel], int(zplane)); src = f'slice {int(zplane)}'
    else:
        b = _maxproj(raw[bf_channel]); src = 'max projection'
    return g.astype(np.float32), r.astype(np.float32), b.astype(np.float32), src


def _bin2(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))


def _nconv(a, m, s):
    return ndi.gaussian_filter(np.where(m, a, 0), s) / np.maximum(ndi.gaussian_filter(m.astype(np.float32), s), 1e-3)


def bf_features(b, m980, patch490=None):
    b2 = _bin2(b)
    far = ~fo.dilate_iter(m980, 40)
    medium = float(np.median(b2[far]))
    rel = b2 / medium
    d = ndi.distance_transform_edt(m980); nd = d / d.max()
    core = m980 & (nd > 0.6); rim = m980 & (nd <= 0.2) & (d > 3)
    out = dict(bf_medium=medium, bf_core_rel=float(np.median(rel[core])), bf_rim_rel=float(np.median(rel[rim])))
    out['bf_core_vs_rim_log2'] = float(np.log2(out['bf_core_rel'] / out['bf_rim_rel']))
    hp = rel - _nconv(rel, m980, 4)
    out['bf_texture_core'] = float(np.std(hp[core])); out['bf_texture_rim'] = float(np.std(hp[rim]))
    out['dark_core_frac'] = float(np.mean(rel[core] < 0.8 * out['bf_rim_rel']))
    # GLCM of the core; level 0 = outside the core (dropped)
    ys, xs = np.nonzero(core); sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    q = (np.clip((rel[sl] - 0.2) / 1.2, 0, 0.999) * 32).astype(np.uint8) + 1; q[~core[sl]] = 0
    P = feature.graycomatrix(q, [2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33, symmetric=True).astype(float)
    P = P[1:, 1:]; P /= np.maximum(P.sum((0, 1), keepdims=True), 1)
    i, j = np.indices(P.shape[:2])
    out['bf_glcm_contrast'] = float(np.mean([(P[:, :, 0, a] * (i - j) ** 2).sum() for a in range(4)]))
    out['bf_glcm_homogeneity'] = float(np.mean([(P[:, :, 0, a] / (1 + (i - j) ** 2)).sum() for a in range(4)]))
    out['bf_glcm_entropy'] = float(np.mean([-(P[:, :, 0, a][P[:, :, 0, a] > 0] * np.log2(P[:, :, 0, a][P[:, :, 0, a] > 0])).sum() for a in range(4)]))
    rel490 = _bin2(rel)
    out['bf_patch_vs_depth_log2'] = np.nan
    if patch490 is not None and patch490.any():
        m490 = _bin2(m980.astype(np.float32)) > 0.5
        d4 = ndi.distance_transform_edt(m490); nd4 = d4 / d4.max(); bins = np.minimum((nd4 * 10).astype(int), 9)
        pin = patch490 & m490; ref = m490 & ~patch490
        num, den = [], []
        for k_ in np.unique(bins[pin]):
            rr = ref & (bins == k_)
            if rr.sum() < 10: continue
            w_ = (pin & (bins == k_)).sum(); num.append(w_ * np.median(rel490[pin & (bins == k_)])); den.append(w_ * np.median(rel490[rr]))
        if num: out['bf_patch_vs_depth_log2'] = float(np.log2(sum(num) / sum(den)))
    return out, rel490.astype(np.float16)


def puncta_features(g, r, mF, bg_g, bg_r):
    ys, xs = np.nonzero(mF)
    sl = (slice(max(ys.min() - 12, 0), ys.max() + 13), slice(max(xs.min() - 12, 0), xs.max() + 13))
    sg, sr, m = g[sl] - bg_g, r[sl] - bg_r, mF[sl]
    d = ndi.distance_transform_edt(m); band = m & (d > 3) & (d <= RIM_UM / PX_FULL_UM)
    rows, peaks = [], []
    for ch, a in (('488', sg), ('405', sr)):
        lg = -ndi.gaussian_laplace(a, LOG_SIGMA) * LOG_SIGMA ** 2
        v = lg[band]; sd = 1.4826 * np.median(np.abs(v - np.median(v)))
        pk = feature.peak_local_max(lg, min_distance=3, threshold_abs=np.median(v) + PUNCTA_K * sd, labels=band.astype(np.uint8))
        peaks += [(int(y), int(x), ch, float((lg[y, x] - np.median(v)) / sd)) for y, x in pk]
    # merge detections within 3 px (keep the stronger)
    peaks.sort(key=lambda p: -p[3]); kept = []
    for p in peaks:
        if not any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 9 for q in kept): kept.append(p)
    Lg, Lr = ndi.gaussian_filter(sg, 8), ndi.gaussian_filter(sr, 8)
    thg = ndi.gaussian_filter(ndi.white_tophat(sg, footprint=_DISK), 1.0); thr_ = ndi.gaussian_filter(ndi.white_tophat(sr, footprint=_DISK), 1.0)
    yy, xx = np.mgrid[-9:10, -9:10]; ring = (np.hypot(yy, xx) >= 5) & (np.hypot(yy, xx) <= 9)
    H, W = sg.shape
    for y, x, ch, snr in kept:
        if y < 9 or x < 9 or y >= H - 9 or x >= W - 9: continue
        win = (slice(y - 9, y + 10), slice(x - 9, x + 10)); core3 = (slice(y - 1, y + 2), slice(x - 1, x + 2))
        rg, rr_ = np.median(sg[win][ring]), np.median(sr[win][ring]); pg, pr = sg[core3].mean(), sr[core3].mean()
        con = float(np.log2((pg / pr) / (rg / rr_))) if min(pg, pr, rg, rr_) > 0 else np.nan
        kind = 'acidic' if con > ACID_PUNCTUM else ('alkaline' if con < -ACID_PUNCTUM else 'neutral')
        rows.append(dict(y=y + sl[0].start, x=x + sl[1].start, detected_in=ch, snr=snr,
                         amp488=float(thg[y, x] / max(Lg[y, x], 1)), amp405=float(thr_[y, x] / max(Lr[y, x], 1)),
                         contrast_log2=con, kind=kind, edge_dist_um=float(d[y, x] * PX_FULL_UM)))
    band_um2 = band.sum() * PX_FULL_UM ** 2
    ok = (ndi.gaussian_filter(sg, 0.8) > 0) & (ndi.gaussian_filter(sr, 0.8) > 0) & band
    lr = np.log(np.maximum(ndi.gaussian_filter(sg, 0.8), 1) / np.maximum(ndi.gaussian_filter(sr, 0.8), 1))
    bp = lr - _nconv(lr, ok, 4)
    n = len(rows); K = [p['kind'] for p in rows]
    out = dict(band_area_um2=band_um2, n_puncta=n, puncta_per_1000um2=n / band_um2 * 1e3 if band_um2 else np.nan,
               frac_acidic_puncta=K.count('acidic') / n if n else np.nan,
               acidic_puncta_per_1000um2=K.count('acidic') / band_um2 * 1e3 if band_um2 else np.nan,
               puncta_contrast_median=float(np.nanmedian([p['contrast_log2'] for p in rows])) if n else np.nan,
               puncta_amp488_median=float(np.median([p['amp488'] for p in rows])) if n else np.nan,
               puncta_amp405_median=float(np.median([p['amp405'] for p in rows])) if n else np.nan,
               rim_cellscale_sd=float(np.std(bp[ok])) if ok.any() else np.nan)
    return out, rows


def texture_well(tp, w, zfiles, raw, zplane, m490, med, floors, alpha, patch490):
    """Worker entry point (lib/parallel.py)."""
    g, r, b, src = load_full(zfiles, raw, zplane, PROJECTION, SEGMENTATION, BF_CHANNEL)
    m980 = ndi.zoom(m490.astype(np.float32), 2, order=1) > 0.5
    mF = ndi.zoom(m490.astype(np.float32), 4, order=1) > 0.5
    fb, bfmap = bf_features(b, m980, patch490)
    bg_g = floors[0] + alpha * (med[0] - floors[0]); bg_r = floors[1] + alpha * (med[1] - floors[1])
    fp, prow = puncta_features(g, r, mF, bg_g, bg_r)
    return dict(bf_source=src, **fb, **fp), prow, bfmap
