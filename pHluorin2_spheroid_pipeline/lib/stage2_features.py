# Stage 2: corrections + spatial features on 490x490 arrays (1 px = 4 full-res px).
# Ratio = 488/405 ; HIGHER = MORE ACIDIC.
import numpy as np, pandas as pd, scipy.ndimage as ndi
from scipy import stats
from skimage import measure
import fastops as fo   # exact faster equivalents (nanmedian along a short axis, cropped dilation)

PX = 4                      # full-res px per array px
TPS = list(PLATE_DIR)   # timepoints in acquisition order (config.PLATE_FOLDERS)
DOSE_UM = {'Oz': [0.01, 0.1, 1], 'Baf': [0.005, 0.02, 0.1], 'Rot': [0.01, 0.1, 1], '3BP': [25, 50, 100],
           'Phlor': [25, 75, 150], 'Olig': [0.03, 0.1, 0.3], 'Ace': [10, 100, 200]}
FULLNAME = {'Oz': 'Ouabain', 'Baf': 'Bafilomycin A1', 'Rot': 'Rotenone', '3BP': '3-Bromopyruvate',
            'Phlor': 'Phloretin', 'Olig': 'Oligomycin', 'Ace': 'Acetazolamide', 'OGM': 'OGM', 'Ogerin': 'Ogerin'}
FULLNAME.update(FULLNAME_CFG or {})

def cond_info(c):
    """-> (compound key, dose_um, dose_rank) ; DMSO -> ('DMSO', 0, 0)."""
    if COND_INFO_CFG and c in COND_INFO_CFG:          # per-experiment table (dose_um may be null = not recorded)
        v = COND_INFO_CFG[c]; return v[0], (np.nan if v[1] is None else float(v[1])), int(v[2])
    if c == 'DMSO': return 'DMSO', 0.0, 0
    if c == 'Fluorobrite': return 'Fluorobrite', 0.0, 0
    if c == '1uM TMZ': return 'TMZ', 1.0, 1
    parts = c.split()
    if parts[0] in DOSE_UM:
        k = 'LMH'.index(parts[1]); return parts[0], DOSE_UM[parts[0]][k], k + 1
    d = float(parts[0].replace('uM', '')); cpd = parts[1]
    ranks = {'OGM': [0.5, 1, 5, 20], 'Ogerin': [0.5, 5, 20, 50]}[cpd]
    return cpd, d, ranks.index(d) + 1

def dose_tag(sub):
    """Legend label for one dose group: the concentration when known, else the strength token ('H'/'L')."""
    um = sub.dose_um.iloc[0]
    if np.isfinite(um): return fmt_dose(um)
    parts = str(sub.condition.iloc[0]).split()
    return parts[-1] if len(parts) > 1 else str(sub.condition.iloc[0])

def fmt_dose(um):
    if not np.isfinite(um): return 'dose not recorded'
    if um == 0: return 'vehicle'
    if um < 1: return f'{um*1000:g} nM'
    return f'{um:g} µM'

# ---------------------------------------------------------------- fields
YY, XX = np.mgrid[:490, :490]; CEN = (YY - 245)**2 + (XX - 245)**2 < 150**2

def _inpaint_field(stack, min_cov=10, grow=4):
    cov = np.isfinite(stack).sum(0); f = fo.nanmedian_small(stack, 0)          # == np.nanmedian(stack, 0)
    hole = ndi.binary_dilation(cov < min_cov, iterations=grow); f[hole] = np.nan
    valid = np.isfinite(f).astype(float)
    num = ndi.gaussian_filter(np.nan_to_num(f), (25, 1.5)); den = ndi.gaussian_filter(valid, (25, 1.5))
    filled = ndi.gaussian_filter(np.where(valid > 0, f, num / np.maximum(den, 1e-6)), 2)
    F = ndi.zoom(filled, 2, order=1)
    return (F / np.median(F[CEN])).astype(np.float32)

def _b2nanmed(a):
    return fo.nanmedian_small(a.reshape(245, 2, 245, 2).transpose(0, 2, 1, 3).reshape(245, 245, 4), 2)   # == np.nanmedian(., axis=2)

def build_fields(G, Rr, M, keys_ok, floors):
    """Per plate: raw ratio flat-field, per-channel total-medium fields (background model shape) and
    per-channel excitation fields (medium minus detector floor), plus per-well medium levels."""
    fields, med = {}, {}
    for tp in TPS:
        fg0, fr0 = floors[tp]
        rs, gs_, rr_, eg, er = [], [], [], [], []
        for (t, w) in keys_ok:
            if t != tp: continue
            g = G[(t, w)].astype(np.float32); r = Rr[(t, w)].astype(np.float32)
            far = ~fo.dilate_iter(M[(t, w)], 20)                  # == ~ndi.binary_dilation(M, iterations=20)
            lg, lr = np.median(g[far]), np.median(r[far]); med[(t, w)] = (lg, lr)
            rs.append(_b2nanmed(np.where(far, g / np.maximum(r, 1), np.nan)) / np.median((g / np.maximum(r, 1))[far]))
            gs_.append(_b2nanmed(np.where(far, g, np.nan)) / lg); rr_.append(_b2nanmed(np.where(far, r, np.nan)) / lr)
            eg.append(_b2nanmed(np.where(far, g - fg0, np.nan)) / (lg - fg0)); er.append(_b2nanmed(np.where(far, r - fr0, np.nan)) / (lr - fr0))
        Eg, Er = _inpaint_field(np.stack(eg)), _inpaint_field(np.stack(er))
        exr = Eg / Er; exr = exr / np.median(exr[CEN])
        fields[tp] = dict(ratio=_inpaint_field(np.stack(rs)), g=_inpaint_field(np.stack(gs_)), r=_inpaint_field(np.stack(rr_)),
                          exratio=exr.astype(np.float32), floor=(fg0, fr0))
    return fields, med

# ---------------------------------------------------------------- per-well maps
def nconv(a, valid, s):
    return ndi.gaussian_filter(np.where(valid, a, 0), s) / np.maximum(ndi.gaussian_filter(valid.astype(float), s), 1e-3)

def ratio_variants(g, r, m, F, medlv, gate_sbr=1.15):
    """raw: no background subtraction (raw-ratio flat-field)
       fs : detector floor subtracted, / excitation-ratio field   (minimum plausible background)
       ms : local medium level x medium field subtracted, / excitation-ratio field (maximum plausible background)
       gate: pixels where both channels exceed the medium model by >= gate_sbr."""
    g = g.astype(np.float32); r = r.astype(np.float32)
    raw = g / np.maximum(r, 1) / F['ratio']
    fg0, fr0 = F['floor']
    fs = (g - fg0) / np.maximum(r - fr0, 1) / F['exratio']
    Bg, Br = medlv[0] * F['g'], medlv[1] * F['r']
    gb, rb = g - Bg, r - Br
    ms = np.where((gb > 0) & (rb > 0), gb / np.maximum(rb, 1e-3), np.nan) / F['exratio']
    a = F.get('alpha', 0.3)
    Cg = fg0 + a * (Bg - fg0); Cr = fr0 + a * (Br - fr0)
    cal = (g - Cg) / np.maximum(r - Cr, 1) / F['exratio']
    gate = m & (g / Bg >= gate_sbr) & (r / Br >= gate_sbr)
    return raw, fs, ms, gate, g / Bg, r / Br, cal

def detrend(R, m, nd, nb=20):
    edges = np.linspace(0, 1, nb + 1); cen = (edges[:-1] + edges[1:]) / 2
    ok = m & np.isfinite(R)
    Rv, nv = R[ok], nd[ok]     # bins selected on the valid-pixel vector: same pixels, same order as R[ok & bin]
    tr = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (nv > a) & (nv <= b)
        tr.append(np.nanmedian(Rv[sel]) if sel.sum() > 5 else np.nan)
    tr = np.array(tr)
    good = np.isfinite(tr)
    trend = np.interp(nd, cen[good], tr[good])
    return R - trend, trend

def corr_length(res, valid, nmax=120):
    z = np.where(valid, res - res[valid].mean(), 0.0); mk = valid.astype(float)
    F = np.fft.rfft2(z, s=(512, 512)); Mk = np.fft.rfft2(mk, s=(512, 512))
    ac = np.fft.irfft2(F * np.conj(F)); nm = np.fft.irfft2(Mk * np.conj(Mk))
    ac = np.fft.fftshift(ac / np.maximum(nm, 1)); ac /= ac[256, 256]
    yy, xx = np.indices(ac.shape); rr = np.hypot(yy - 256, xx - 256).astype(int)
    prof = np.bincount(rr.ravel(), ac.ravel()) / np.bincount(rr.ravel())
    below = np.where(prof[1:nmax] < 1 / np.e)[0]
    return float((below[0] + 1) * PX) if len(below) else np.nan

def moran_I(res, valid):
    z = np.where(valid, res - res[valid].mean(), 0.0); num = 0.0; W = 0
    for dy, dx in [(0, 1), (1, 0)]:
        a = z[:z.shape[0]-dy, :z.shape[1]-dx]; b = z[dy:, dx:]
        v = valid[:z.shape[0]-dy, :z.shape[1]-dx] & valid[dy:, dx:]
        num += 2 * (a * b)[v].sum(); W += 2 * v.sum()
    n = valid.sum()
    return float((n / W) * num / (z[valid]**2).sum()) if W and n > 10 else np.nan

EDGE = 3          # array px excluded at edge (12 full-res px)
RIM = 15          # rim band width (60 full-res px)
NDB = np.linspace(0, 1, 6)          # normalised-depth bins for patch location (0 = edge)
ABSB = np.array([0, 10, 20, 40, 80, 999])   # array px -> 0-40, 40-80, 80-160, 160-320, >320 full px
PROFB = np.arange(0, 80, 3)          # abs profile, 12 full px steps

PATCH_LOGTHR = float(np.log(1.06))

def patches(res_s, valid, thr, minpx=10):
    L = measure.label(valid & (res_s > thr))
    cnt = np.bincount(L.ravel()); ok = cnt >= minpx; ok[0] = False
    return ok[L]

def well_features(g, r, m, F, medlv, shared=None):
    out, maps = {}, {}
    d = ndi.distance_transform_edt(m); dmax = d.max(); nd = d / dmax
    p = measure.regionprops(m.astype(np.uint8))[0]
    out.update(area_px=p.area * PX**2, eq_diam_px=p.equivalent_diameter_area * PX,
               circularity=4*np.pi*p.area / p.perimeter**2, solidity=p.solidity)
    cont = max(measure.find_contours(m.astype(float), 0.5), key=len)
    rad = np.hypot(cont[:, 0] - p.centroid[0], cont[:, 1] - p.centroid[1]); out['edge_roughness'] = rad.std() / rad.mean()
    raw, fs, ms, gate, sbr_g, sbr_r, cal = ratio_variants(g, r, m, F, medlv)
    valid = m & (d > EDGE)
    out.update(g_mean=float(g[m].astype(np.float32).mean()), r_mean=float(r[m].astype(np.float32).mean()),
               g_medium=float(medlv[0]), r_medium=float(medlv[1]), medium_ratio=float(medlv[0] / medlv[1]),
               sbr488=float(np.median(sbr_g[m])), sbr405=float(np.median(sbr_r[m])),
               gated_frac=float(gate[valid].mean()))
    core = nd > 0.6
    for tag, R, V in [('', raw, valid), ('_fs', fs, valid), ('_cal', cal, valid), ('_ms', ms, valid & gate & np.isfinite(ms))]:
        if V.sum() < 200:
            continue
        rim = V & (d <= EDGE + RIM); cr = V & core
        out['ratio_mean' + tag] = float(np.nanmean(R[V])); out['ratio_median' + tag] = float(np.nanmedian(R[V]))
        out['rim' + tag] = float(np.nanmedian(R[rim])) if rim.sum() > 20 else np.nan
        out['core' + tag] = float(np.nanmedian(R[cr])) if cr.sum() > 20 else np.nan
        out['rim_minus_core' + tag] = out['rim' + tag] - out['core' + tag]
        res, trend = detrend(np.where(V, R, np.nan), V, nd)
        if shared is not None and tag in shared:
            res_shared_removed = res - np.nan_to_num(shared[tag])
        else:
            res_shared_removed = res
        rs = nconv(np.nan_to_num(res_shared_removed), V, 1.5)
        rel = nconv(np.nan_to_num(res_shared_removed / trend), V, 1.5)
        # ---- acidic/alkaline PATCHES (log-space definition, adopted 2026-10-20). The earlier definition (residual
        # after additive shared-pattern removal > +0.06 ratio units) is biased by the spheroid's mean level: the
        # shared pattern carries the amplitude of a ratio~1 spheroid, so it leaves an offset on spheroids far from
        # the plate level, and an absolute threshold is a different relative excursion at different levels.
        # Now: detrend log(R) against its own radial profile, remove the plate's shared pattern in log space,
        # re-centre on the spheroid's own median, patch = >= 10 px connected region > log(1.06), i.e. 6 % more
        # acidic (or alkaline) than the local radial expectation. Old values kept as *_abs columns.
        lres, _ = detrend(np.where(V, np.log(np.maximum(R, 1e-3)), np.nan), V, nd)
        if shared is not None and (tag + '_log') in shared:
            lres = lres - np.nan_to_num(shared[tag + '_log'])
        ls = nconv(np.nan_to_num(lres), V, 1.5); ls = ls - np.median(ls[V])
        pa_abs = patches(rs, V, 0.06); pk_abs = patches(-rs, V, 0.06)
        out['resid_sd' + tag] = float(np.nanstd(res_shared_removed[V]))
        out['resid_sd_rel' + tag] = float(np.nanstd((res_shared_removed / trend)[V]))
        pa = patches(ls, V, PATCH_LOGTHR); pk = patches(-ls, V, PATCH_LOGTHR); pr = patches(rel, V, 0.04)
        out['acid_patch_frac' + tag] = float(pa[V].mean()); out['alk_patch_frac' + tag] = float(pk[V].mean())
        out['acid_patch_frac_abs' + tag] = float(pa_abs[V].mean()); out['alk_patch_frac_abs' + tag] = float(pk_abs[V].mean())
        out['log_resid_mad' + tag] = float(1.4826 * np.median(np.abs(ls[V] - np.median(ls[V]))))
        out['acid_patch_frac_rel' + tag] = float(pr[V].mean())
        out['corr_len_px' + tag] = corr_length(np.where(V, res_shared_removed, 0), V)
        if tag in ('', '_cal'):
            out['moran_I' + tag] = moran_I(np.nan_to_num(res_shared_removed), V)
            # pixel-scale noise: high-pass residual
            hp = R - nconv(R, V, 1.0)
            out['pixel_noise_sd' + tag] = float(np.nanstd(hp[V & (d > EDGE + 2)]))
            # radial location of patches
            for i, (a, b) in enumerate(zip(NDB[:-1], NDB[1:])):
                band = V & (nd > a) & (nd <= b)
                out[f'patch_occ_nd{i}' + tag] = float(pa[band].mean()) if band.sum() > 10 else np.nan
                out[f'pos_excess_nd{i}' + tag] = float(np.clip(ls[band], 0, None).mean()) if band.sum() > 10 else np.nan
            for i, (a, b) in enumerate(zip(ABSB[:-1], ABSB[1:])):
                band = V & (d > max(a, EDGE)) & (d <= b)
                out[f'patch_occ_abs{i}' + tag] = float(pa[band].mean()) if band.sum() > 10 else np.nan
            occ = pa[V].mean()
            outer = V & (nd <= 0.2)
            out['patch_boundary_enrich' + tag] = float(pa[outer].mean() / occ) if occ > 0 else np.nan
            # patch centroid depth (normalised) vs pixel-average depth
            out['patch_depth_mean' + tag] = float(nd[pa].mean()) if pa.any() else np.nan
            out['pixel_depth_mean' + tag] = float(nd[V].mean())
            # brightness vs spectral decomposition (log space, radially detrended, smoothed)
            off = ((F['floor'][0] + F.get('alpha', 0.3) * (medlv[0] * F['g'] - F['floor'][0])),
                   (F['floor'][1] + F.get('alpha', 0.3) * (medlv[1] * F['r'] - F['floor'][1]))) if tag == '_cal' else (0, 0)
            lg = np.log(np.maximum(g.astype(np.float32) - off[0], 1)); lr = np.log(np.maximum(r.astype(np.float32) - off[1], 1))
            rg, _ = detrend(np.where(V, lg, np.nan), V, nd); rr_, _ = detrend(np.where(V, lr, np.nan), V, nd)
            rg = nconv(np.nan_to_num(rg), V, 1.5); rr_ = nconv(np.nan_to_num(rr_), V, 1.5)
            spec, bright = (rg - rr_)[V], (rg + rr_)[V]
            out['corr_spec_bright' + tag] = float(np.corrcoef(spec, bright)[0, 1])
            out['corr_488_405_resid' + tag] = float(np.corrcoef(rg[V], rr_[V])[0, 1])
            out['patch_bright_resid' + tag] = float(np.mean((rg + rr_)[pa])) if pa.any() else np.nan
            out['bright_resid_sd' + tag] = float(np.std(bright))
            # profiles
            Rv, dv, ndv = R[V], d[V], nd[V]     # same pixels, same order as R[V & band]
            out['_prof_abs' + tag] = np.array([np.nanmedian(Rv[(dv > a) & (dv <= a + 3)]) if ((dv > a) & (dv <= a + 3)).sum() > 15 else np.nan
                                         for a in PROFB])
            out['_prof_nd' + tag] = np.array([np.nanmedian(Rv[(ndv > a) & (ndv <= b)]) for a, b in
                                        zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:])])
            maps[tag] = dict(R=np.where(m, R, np.nan).astype(np.float16),
                        res=np.where(V, rs, np.nan).astype(np.float16),
                        lres=np.where(V, ls, np.nan).astype(np.float16),
                        lres_raw=np.where(V, lres, np.nan).astype(np.float16) if shared is None else None,
                        patch=pa, centroid=p.centroid, eq_r=p.equivalent_diameter_area / 2)
    return out, maps


# ---------------------------------------------------------------- worker entry point (lib/parallel.py)
def well_features_job(key, g, r, m, medlv, pass_no):
    """well_features for one spheroid in a worker process. Plate fields (PAR_FIELDS) and, for pass 2, the plate's
    shared residual pattern (PAR_SHARED) are sent to each worker once when the pool starts.
    pass 0 returns the calibrated maps only (step 7); pass 1 the residual maps needed for the shared pattern;
    pass 2 (features, maps)."""
    tp = key[0]
    if pass_no == 0:          # step 7: calibrated maps of the max projection, no shared-pattern removal
        return well_features(g, r, m, PAR_FIELDS[tp], medlv)[1]['_cal']
    if pass_no == 1:
        _, mp = well_features(g, r, m, PAR_FIELDS[tp], medlv)
        return {tg: (mp[tg]['res'], mp[tg]['lres_raw']) for tg in ['', '_cal']}
    return well_features(g, r, m, PAR_FIELDS[tp], medlv, shared=PAR_SHARED[tp])
