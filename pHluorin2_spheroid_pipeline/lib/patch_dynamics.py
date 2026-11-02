# Acid-patch dynamics across timepoints (same well = same spheroid): movement, growth relative to the spheroid,
# number of patches and patch emergence / loss. Works on the step-4 maps (spatial_maps_cal.pkl): a patch is a
# connected region of >= 10 px where the log residual (lres) exceeds log(1.06), exactly as defined in step 4.
#
# Common frame. Every spheroid-timepoint is expressed in its own centred frame scaled by its equivalent radius
# (u, v in units of radius; u = down, v = right). Drift of the spheroid in the well and isotropic growth therefore
# do not register as patch movement. Rotation of the whole spheroid between reads is handled two ways:
#   'fixed'  no rotation (assumes the spheroid kept its orientation)
#   'rotreg' the rotation that best correlates the two log-residual maps (5 deg grid, then 1 deg) is removed first.
#            This search also aligns unrelated maps a little, so its chance level (random pairs, same search) is higher.
#
# Linking patches between two timepoints: one-to-one assignment on centroid distance (Hungarian), accepted when
# <= LINK_MAX radii -> 'persist'. Unlinked patches are then classified by overlap in the common frame:
#   later patch with >= OVL_MIN of its area on an earlier patch -> 'split';  otherwise -> 'emerged'
#   earlier patch with >= OVL_MIN of its area on a later patch  -> 'merged'; otherwise -> 'vanished'
#   (a new patch that only touches a neighbour's outline after registration is 'emerged', not a split)
#
# Patch units: neighbouring acid patches fuse into one connected region at the log(1.06) threshold when they are
#              closer than ~4 blob widths. By default (SPHEROID_PATCH_SPLIT=watershed) each connected patch region is
#              split at its separate peaks: watershed of the log residual seeded by maxima at least PEAK_H (0.04 log
#              units, ~4 % acidity) more prominent than their surroundings; pieces < MIN_PX are folded into a neighbour.
#              The union of patch pixels (patch area) is unchanged. SPHEROID_PATCH_SPLIT=components counts plain
#              connected components, as in step 4.
# Movement   : centroid displacement of persisting patches (radii and um), split into radial and tangential parts.
#              Jaccard distance of the two patch masks (1 - |A n B| / |A u B|, inside the region valid on both days)
#              and centroid-set Jaccard distance (centroids within JACC_TOL radii count as the same patch:
#              1 - m / (n1 + n2 - m)), each with its chance level.
# Growth     : log2 fold-change of patch area minus log2 fold-change of spheroid area (0 = patch grows in proportion
#              to the spheroid; +1 = patch area doubles relative to the spheroid), per persisting patch and in total.
# Number     : patches per spheroid, density per mm2, emerged / vanished per day, turnover.
# Patch acidity and brightness (per patch, patch_table.csv):
#   ratio_mean / ratio_median / ratio_p95   calibrated 488/405 inside the patch (step-4 map R; higher = more acidic).
#        Patches are selected for being more acidic than their surroundings, so these are high by construction:
#        compare them between conditions, not with the rest of the spheroid.
#   ratio_ring        median ratio of non-patch tissue in a RING_PX-wide ring around the patch
#   ratio_depthref    median ratio of non-patch tissue at the same normalised depth (+/- DEPTH_TOL)
#   log_contrast_ring / log_contrast_depth   ln(patch median / reference): how much more acidic than the local or
#        same-depth tissue (depth-matched contrast is the least sensitive to the rim/core optical gradient)
#   lres_mean / lres_peak   the step-4 log residual (excess over the radial expectation) used to define the patch
#   sig488_log2_vs_depth / sig405_log2_vs_depth   medium-subtracted brightness of each channel in the patch vs
#        non-patch tissue at the same depth (log2); *_vs_ring: vs the surrounding ring.
#        A pH drop moves the two channels apart (488/405 up; for ratiometric pHluorin2 the 405-excited signal falls
#        and the 488-excited signal does not). Losing signal (fewer cells, necrosis, shadowing) dims BOTH, and with
#        residual medium in the calibrated model that alone reads as 'acidic'.
#   signal_loss       both channels >= 20 % dimmer than same-depth tissue: probably not a pH patch. Flagged, not
#        removed. (The ring is not used for this: a dim spot also dims its own surroundings.)
#   eq_diam_um, eccentricity, solidity, dist_edge_um (centroid to spheroid edge), depth (normalised, 0 = edge)
import os, numpy as np, pandas as pd, scipy.ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import measure, morphology, segmentation

PXUM_ARR = 4 * 1367 / 1960          # um per 490x490 array px (= 4 full-res px)
LINK_MAX = 0.35                     # max centroid distance for a one-to-one link (radii)
JACC_TOL = 0.15                     # centroid-set Jaccard tolerance (radii)
OVL_MIN = 0.3                       # overlap fraction for split / merge
RING_PX = 4                         # reference ring around a patch (array px = 11 um)
DEPTH_TOL = 0.05                    # same-depth reference band (normalised depth)
SIGNAL_LOSS = np.log2(0.8)          # both channels this much dimmer than the ring -> signal_loss
SPLIT = os.environ.get('SPHEROID_PATCH_SPLIT', 'watershed')
PEAK_H = float(os.environ.get('SPHEROID_PATCH_PEAK_H', 0.04))   # min peak prominence (log units) for a separate patch
MIN_PX = 10                         # as in step 4
NG = 160; _g = np.linspace(-1.3, 1.3, NG); GU, GV = np.meshgrid(_g, _g, indexing='ij')
GRID_DA = (_g[1] - _g[0]) ** 2      # area of one grid cell (radius^2)


# ------------------------------------------------------------------ one spheroid-timepoint
def patch_labels(lres, patch):
    """Label map of patch units (see header). lres: smoothed log residual (NaN outside the analysed region)."""
    if SPLIT != 'watershed' or not patch.any():
        return measure.label(patch)
    f = np.where(np.isfinite(lres), lres, -1).astype(np.float64)
    peaks = morphology.h_maxima(np.where(patch, f, -1), PEAK_H).astype(bool) & patch
    comp = measure.label(patch)
    missing = np.setdiff1d(np.unique(comp[patch]), np.unique(comp[peaks]))
    for c in missing:                                   # every region keeps at least its highest point as a seed
        idx = np.flatnonzero((comp == c).ravel()); peaks.flat[idx[np.argmax(f.flat[idx])]] = True
    markers = measure.label(peaks)
    lab = segmentation.watershed(-f, markers, mask=patch)
    for _ in range(3):                                  # fold pieces < MIN_PX into the neighbour they touch most
        sizes = np.bincount(lab.ravel()); small = [k for k in range(1, len(sizes)) if 0 < sizes[k] < MIN_PX]
        if not small: break
        for k in small:
            ring = ndi.binary_dilation(lab == k) & (lab != k) & (lab > 0)
            if ring.any(): lab[lab == k] = np.bincount(lab[ring]).argmax()
    return segmentation.relabel_sequential(lab)[0]


PATCH_COLS = ['label', 'area_px', 'area_um2', 'eq_diam_um', 'eccentricity', 'solidity', 'u', 'v', 'depth', 'dist_edge_um',
              'lres_mean', 'lres_peak', 'ratio_mean', 'ratio_median', 'ratio_p95', 'ratio_ring', 'ratio_depthref',
              'log_contrast_ring', 'log_contrast_depth', 'sig488_log2_vs_depth', 'sig405_log2_vs_depth',
              'sig488_log2_vs_ring', 'sig405_log2_vs_ring', 'signal_loss']


def alkaline_mask(lres, V, thr=float(np.log(1.06)), minpx=MIN_PX):
    """Alkaline patches exactly as step 4 defines them (patches(-ls, V, log 1.06)), rebuilt from the stored log
    residual. The stored map is float16, so a pixel lying within float16 rounding of the threshold can differ."""
    Lb = measure.label(V & (-lres > thr)); cnt = np.bincount(Lb.ravel()); ok = cnt >= minpx; ok[0] = False
    return ok[Lb]


def spheroid_patches(mp, g=None, r=None, med=None, polarity='acid'):
    """Patch table and frame of one spheroid-timepoint from its step-4 map dict (R, lres, patch, centroid, eq_r).
    g, r: the 490x490 488/405 arrays of the same spheroid-timepoint (stage1_arrays.npz) and med = (488, 405) medium
    levels, for the per-patch brightness check; without them those columns are NaN.
    polarity 'alk': alkaline patches (log residual below -log 1.06); contrasts are then negative."""
    lres = mp['lres'].astype(np.float32); V = np.isfinite(lres); R = mp['R'].astype(np.float32); m = np.isfinite(R)
    cy, cx = mp['centroid']; rr = float(mp['eq_r'])
    d = ndi.distance_transform_edt(m); nd = d / d.max()
    sign = 1.0 if polarity == 'acid' else -1.0
    pmask = (mp['patch'] if polarity == 'acid' else alkaline_mask(lres, V)) & V
    L = patch_labels(sign * lres, pmask)
    anyp = L > 0; ref_ok = V & ~anyp
    sg = sr = None
    if g is not None:
        sg = g.astype(np.float32) - med[0]; sr = r.astype(np.float32) - med[1]
    rows = []
    for p in measure.regionprops(L, intensity_image=sign * lres):
        y, x = p.centroid; yi, xi = int(round(y)), int(round(x))
        sl = tuple(slice(max(0, a - RING_PX - 1), b + RING_PX + 1) for a, b in zip(p.bbox[:2], p.bbox[2:]))
        own = L[sl] == p.label
        ring = ndi.binary_dilation(own, iterations=RING_PX) & ~own & ref_ok[sl]
        pv = R[sl][own]; pv = pv[np.isfinite(pv)]; rv = R[sl][ring]
        ndp = float(nd[sl][own].mean())
        band = ref_ok & (np.abs(nd - ndp) <= DEPTH_TOL)
        rm = float(np.median(pv)) if len(pv) else np.nan
        ring_med = float(np.median(rv)) if ring.sum() >= 10 else np.nan
        dep_med = float(np.median(R[band])) if band.sum() >= 20 else np.nan
        rec = dict(label=p.label, area_px=int(p.area), area_um2=p.area * PXUM_ARR ** 2,
                   eq_diam_um=p.equivalent_diameter_area * PXUM_ARR, eccentricity=p.eccentricity, solidity=p.solidity,
                   u=(y - cy) / rr, v=(x - cx) / rr, depth=float(nd[yi, xi]), dist_edge_um=float(d[yi, xi] * PXUM_ARR),
                   lres_mean=sign * float(p.intensity_mean), lres_peak=sign * float(p.intensity_max),
                   ratio_mean=float(pv.mean()) if len(pv) else np.nan, ratio_median=rm,
                   ratio_p95=float(np.percentile(pv, 95)) if len(pv) else np.nan,
                   ratio_ring=ring_med, ratio_depthref=dep_med,
                   log_contrast_ring=float(np.log(rm / ring_med)) if rm > 0 and ring_med > 0 else np.nan,
                   log_contrast_depth=float(np.log(rm / dep_med)) if rm > 0 and dep_med > 0 else np.nan,
                   sig488_log2_vs_depth=np.nan, sig405_log2_vs_depth=np.nan, sig488_log2_vs_ring=np.nan,
                   sig405_log2_vs_ring=np.nan, signal_loss=False)
        if sg is not None:
            for ch, a in (('488', sg), ('405', sr)):
                pin = float(np.median(a[sl][own]))
                for ref, ok_ in (('depth', band.sum() >= 20), ('ring', ring.sum() >= 10)):
                    if not ok_: continue
                    pout = float(np.median(a[band] if ref == 'depth' else a[sl][ring]))
                    rec[f'sig{ch}_log2_vs_{ref}'] = float(np.log2(pin / pout)) if pin > 0 and pout > 0 else np.nan
            rec['signal_loss'] = bool(rec['sig488_log2_vs_depth'] < SIGNAL_LOSS and rec['sig405_log2_vs_depth'] < SIGNAL_LOSS)
        rows.append(rec)
    T = pd.DataFrame(rows, columns=PATCH_COLS)
    summ = dict(n_patches=len(T), valid_area_um2=float(V.sum() * PXUM_ARR ** 2),
                spheroid_area_um2=float(m.sum() * PXUM_ARR ** 2), patch_area_um2=float(T.area_um2.sum()),
                eq_r_um=rr * PXUM_ARR)
    summ['patch_frac'] = summ['patch_area_um2'] / summ['valid_area_um2'] if summ['valid_area_um2'] else np.nan
    summ['patch_density_per_mm2'] = summ['n_patches'] / summ['valid_area_um2'] * 1e6 if summ['valid_area_um2'] else np.nan
    summ['mean_patch_area_um2'] = float(T.area_um2.mean()) if len(T) else np.nan
    summ['mean_patch_lres'] = float(T.lres_mean.mean()) if len(T) else np.nan
    w_ = T.area_px.values
    summ['patch_ratio_mean'] = float(np.average(T.ratio_mean, weights=w_)) if len(T) and np.isfinite(T.ratio_mean).all() else np.nan
    summ['nonpatch_ratio_median'] = float(np.median(R[ref_ok])) if ref_ok.any() else np.nan
    summ['patch_contrast_depth_mean'] = float(T.log_contrast_depth.mean()) if len(T) else np.nan
    summ['patch_contrast_ring_mean'] = float(T.log_contrast_ring.mean()) if len(T) else np.nan
    summ['n_signal_loss'] = int(T.signal_loss.sum()) if len(T) else 0
    summ['frac_signal_loss'] = float(T.signal_loss.mean()) if len(T) else np.nan
    return T, summ, L


def resample(mp, L, theta_deg=0.0):
    """Sample one spheroid's label map, validity and lres onto the common normalised grid (optionally rotated)."""
    th = np.deg2rad(theta_deg); c, s = np.cos(th), np.sin(th)
    cy, cx = mp['centroid']; r = float(mp['eq_r'])
    yy = cy + r * (GU * c - GV * s); xx = cx + r * (GU * s + GV * c)
    lab = ndi.map_coordinates(L, [yy, xx], order=0, cval=0)
    lres = ndi.map_coordinates(np.nan_to_num(mp['lres'].astype(np.float32), nan=np.nan), [yy, xx], order=1, cval=np.nan)
    return lab, np.isfinite(lres), lres


def to_frame(u, v, theta_deg):
    """Coordinates of a later spheroid's patches, (u, v) in its own frame, expressed in the grid frame used when it
    was resampled with rotation theta (the inverse of the sampling rotation)."""
    th = np.deg2rad(theta_deg); c, s = np.cos(th), np.sin(th)
    return u * c + v * s, -u * s + v * c


def best_rotation(gA, mpB, LB):
    """Rotation (deg) of spheroid B that best correlates its log-residual map with A's (already resampled, gA)."""
    def score(t):
        _, okB, lB = resample(mpB, LB, t); ok = gA[1] & okB
        return np.corrcoef(gA[2][ok], lB[ok])[0, 1] if ok.sum() > 200 else -2
    coarse = np.arange(0, 360, 5); sc = [score(t) for t in coarse]; t0 = coarse[int(np.argmax(sc))]
    fine = np.arange(t0 - 4, t0 + 5); sc = [score(t) for t in fine]
    return float(fine[int(np.argmax(sc))] % 360), float(max(sc))


# ------------------------------------------------------------------ one transition
def transition(mpA, TA, sA, LA, mpB, TB, sB, LB, register='fixed', gA=None):
    gA = gA if gA is not None else resample(mpA, LA)
    theta, rot_r = (0.0, np.nan) if register == 'fixed' else best_rotation(gA, mpB, LB)
    labB, okB, _ = resample(mpB, LB, theta); labA, okA = gA[0], gA[1]; ok = okA & okB
    # mask Jaccard inside the region valid on both days
    a, b = (labA > 0) & ok, (labB > 0) & ok
    inter, union = int((a & b).sum()), int((a | b).sum())
    fA, fB = a.sum() / max(ok.sum(), 1), b.sum() / max(ok.sum(), 1)
    out = dict(rotation_deg=theta, rotation_r=rot_r,
               jaccard_mask_dist=1 - inter / union if union else np.nan,
               jaccard_mask_chance=1 - (fA * fB / (fA + fB - fA * fB)) if (fA + fB) > 0 else np.nan)
    # centroids of B in A's frame
    uA, vA = TA.u.values, TA.v.values; uB, vB = to_frame(TB.u.values, TB.v.values, theta)
    n1, n2 = len(TA), len(TB)
    links = []
    if n1 and n2:
        D = np.hypot(uA[:, None] - uB[None, :], vA[:, None] - vB[None, :])
        ri, ci = linear_sum_assignment(np.where(D <= LINK_MAX, D, 1e6))
        pairs = [(i, j) for i, j in zip(ri, ci) if D[i, j] <= LINK_MAX]
        rt, ct = linear_sum_assignment(np.where(D <= JACC_TOL, D, 1e6))
        m_tol = int(sum(D[i, j] <= JACC_TOL for i, j in zip(rt, ct)))
        # overlap between labels in the common frame
        la, lb = labA[ok], labB[ok]; both = (la > 0) & (lb > 0)
        O = np.zeros((n1 + 1, n2 + 1), int)              # labels are 1..n in table order (skimage regionprops)
        np.add.at(O, (la[both], lb[both]), 1); O = O[1:, 1:]
        gA_area = np.bincount(la, minlength=n1 + 1)[1:]; gB_area = np.bincount(lb, minlength=n2 + 1)[1:]
    else:
        pairs, m_tol, O = [], 0, np.zeros((n1, n2), int); gA_area = np.zeros(n1); gB_area = np.zeros(n2)
    out['centroid_jaccard_dist'] = 1 - m_tol / (n1 + n2 - m_tol) if (n1 + n2) else np.nan
    linkedA = {i for i, _ in pairs}; linkedB = {j for _, j in pairs}
    logS = np.log2(sB['spheroid_area_um2'] / sA['spheroid_area_um2'])
    for i, j in pairs:
        rA, rB = np.hypot(uA[i], vA[i]), np.hypot(uB[j], vB[j])
        dth = np.degrees(np.angle(np.exp(1j * (np.arctan2(uB[j], vB[j]) - np.arctan2(uA[i], vA[i])))))
        disp = float(np.hypot(uB[j] - uA[i], vB[j] - vA[i]))
        # absolute displacement relative to the spheroid centre (um): no scaling for growth
        pa = np.array([uA[i], vA[i]]) * sA['eq_r_um']; pb = np.array([uB[j], vB[j]]) * sB['eq_r_um']
        lg = np.log2(TB.area_um2.values[j] / TA.area_um2.values[i])
        links.append(dict(event='persist', labelA=int(TA.label.values[i]), labelB=int(TB.label.values[j]),
                          uA=uA[i], vA=vA[i], uB=uB[j], vB=vB[j], disp=disp, radial_shift=rB - rA,
                          tangential=float(np.sqrt(max(disp ** 2 - (rB - rA) ** 2, 0))), dtheta_deg=float(dth),
                          disp_um=float(np.hypot(*(pb - pa))), depth_shift=TB.depth.values[j] - TA.depth.values[i],
                          areaA_um2=TA.area_um2.values[i], areaB_um2=TB.area_um2.values[j], log2fc=lg, rel_log2fc=lg - logS,
                          lresA=TA.lres_mean.values[i], lresB=TB.lres_mean.values[j],
                          contrast_change=TB.log_contrast_depth.values[j] - TA.log_contrast_depth.values[i],
                          ratio_log2fc=float(np.log2(TB.ratio_median.values[j] / TA.ratio_median.values[i]))))
    for j in range(n2):
        if j in linkedB: continue
        src = np.flatnonzero(O[:, j]) if n1 else []
        ev = 'split' if len(src) and O[src, j].sum() >= OVL_MIN * max(gB_area[j], 1) else 'emerged'
        links.append(dict(event=ev, labelA=int(TA.label.values[src[np.argmax(O[src, j])]]) if ev == 'split' else -1,
                          labelB=int(TB.label.values[j]), uB=uB[j], vB=vB[j], areaB_um2=TB.area_um2.values[j]))
    for i in range(n1):
        if i in linkedA: continue
        dst = np.flatnonzero(O[i, :]) if n2 else []
        ev = 'merged' if len(dst) and O[i, dst].sum() >= OVL_MIN * max(gA_area[i], 1) else 'vanished'
        links.append(dict(event=ev, labelA=int(TA.label.values[i]),
                          labelB=int(TB.label.values[dst[np.argmax(O[i, dst])]]) if ev == 'merged' else -1,
                          uA=uA[i], vA=vA[i], areaA_um2=TA.area_um2.values[i]))
    Lk = pd.DataFrame(links)
    ev = Lk.event.value_counts() if len(Lk) else pd.Series(dtype=int)
    P = Lk[Lk.event == 'persist'] if len(Lk) else Lk
    out.update(n1=n1, n2=n2, **{f'n_{e}': int(ev.get(e, 0)) for e in ['persist', 'emerged', 'vanished', 'split', 'merged']},
               disp_median=float(P.disp.median()) if len(P) else np.nan, disp_mean=float(P.disp.mean()) if len(P) else np.nan,
               disp_um_median=float(P.disp_um.median()) if len(P) else np.nan,
               radial_shift_mean=float(P.radial_shift.mean()) if len(P) else np.nan,
               tangential_mean=float(P.tangential.mean()) if len(P) else np.nan,
               spheroid_log2fc=float(logS),
               patch_area_log2fc=float(np.log2(sB['patch_area_um2'] / sA['patch_area_um2'])) if sA['patch_area_um2'] > 0 and sB['patch_area_um2'] > 0 else np.nan,
               patch_log2fc_mean=float(P.log2fc.mean()) if len(P) else np.nan,
               rel_growth_patch_mean=float(P.rel_log2fc.mean()) if len(P) else np.nan,
               contrast_change_mean=float(P.contrast_change.mean()) if len(P) else np.nan,
               ratio_log2fc_mean=float(P.ratio_log2fc.mean()) if len(P) else np.nan)
    out['rel_growth_total'] = out['patch_area_log2fc'] - logS
    out['turnover'] = (out['n_emerged'] + out['n_vanished']) / (n1 + n2) if (n1 + n2) else np.nan
    return out, Lk


def per_day(df):
    """Rates per day for a transition table with a dt_h column."""
    days = df.dt_h / 24.0
    for c_in, c_out in [('n_emerged', 'emerged_per_day'), ('n_vanished', 'vanished_per_day'), ('disp_median', 'disp_per_day'),
                        ('disp_um_median', 'disp_um_per_day'), ('rel_growth_patch_mean', 'rel_growth_patch_per_day'),
                        ('rel_growth_total', 'rel_growth_total_per_day')]:
        df[c_out] = df[c_in] / days
    return df
