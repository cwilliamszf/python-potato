# Per-plane (single focal plane) analysis of 488/405 (HIGHER = MORE ACIDIC), calibrated background model.
import numpy as np, scipy.ndimage as ndi
import fastops as fo
from scipy import stats

# values estimated in step 4 (calibration.json); these defaults are the ones obtained in the original run
FLOORS = {'24h': (6564, 5459), '48h': (6590, 5879), '72h': (6306, 6017)}
ALPHA = {'24h': 0.32, '48h': 0.32, '72h': 0.22}
NDB10 = np.linspace(0, 1, 11)

def plane_medium(a, m):
    far = ~fo.dilate_iter(m, 20)                      # == ~ndi.binary_dilation(m, iterations=20)
    return float(np.median(a[far]))

def cal_ratio(g, r, F, lg, lr):
    fg0, fr0 = F['floor']; a = F['alpha']
    Cg = fg0 + a * (lg * F['g'] - fg0); Cr = fr0 + a * (lr * F['r'] - fr0)
    sg, sr = g - Cg, r - Cr
    return sg / np.maximum(sr, 1) / F['exratio'], sg, sr

def edge_sharpness(img, m, band=4):
    d_in = ndi.distance_transform_edt(m); d_out = ndi.distance_transform_edt(~m)
    ring = ((d_in > 0) & (d_in <= band)) | ((d_out > 0) & (d_out <= band))
    gy, gx = np.gradient(ndi.gaussian_filter(img, 1.0))
    return float(np.mean(np.hypot(gy, gx)[ring]))

def lap_focus(img, m):
    return float(np.var(ndi.laplace(ndi.gaussian_filter(img, 1.0))[m]))

def plane_profiles(stk, m, F):
    """Return dict of per-plane arrays: ratio/intensity radial profiles + focus metrics."""
    nz = stk['g'].shape[0]
    d = ndi.distance_transform_edt(m); nd = d / d.max()
    bins = [(m & (nd > a) & (nd <= b)) for a, b in zip(NDB10[:-1], NDB10[1:])]
    out = {k: np.full((nz, 10), np.nan) for k in ['R', 'sg', 'sr']}
    for k in ['lg', 'lr', 'focus_bf', 'focus_g', 'edge_g', 'edge_r', 'edge_bf', 'Rmed', 'rim_core']:
        out[k] = np.full(nz, np.nan)
    for z in range(nz):
        g = stk['g'][z].astype(np.float32); r = stk['r'][z].astype(np.float32); b = stk['b'][z].astype(np.float32)
        lg, lr = plane_medium(g, m), plane_medium(r, m)
        R, sg, sr = cal_ratio(g, r, F, lg, lr)
        for i, bm in enumerate(bins):
            out['R'][z, i] = np.median(R[bm]); out['sg'][z, i] = np.median(sg[bm]); out['sr'][z, i] = np.median(sr[bm])
        out['lg'][z], out['lr'][z] = lg, lr
        out['focus_bf'][z] = lap_focus(b, m); out['focus_g'][z] = lap_focus(g, m)
        out['edge_g'][z] = edge_sharpness(g, m); out['edge_r'][z] = edge_sharpness(r, m); out['edge_bf'][z] = edge_sharpness(b, m)
        out['Rmed'][z] = np.median(R[m & (d > 3)])
        out['rim_core'][z] = np.median(R[m & (nd <= 0.2) & (d > 3)]) - np.median(R[m & (nd > 0.6)])
    return out


# chromatic focal shift between the 488 and 405 stacks (slices): high-pass correlation search
def hp(a):
    a = a.astype(np.float32); return ndi.gaussian_filter(a, 1.0) - ndi.gaussian_filter(a, 6.0)

def chroma_shift(stk, m, zmax=14, dmax=9):
    mm = fo.dilate_iter(m, 6)                          # == ndi.binary_dilation(m, iterations=6)
    G_ = [hp(stk['g'][z])[mm] for z in range(zmax + dmax)]; R_ = [hp(stk['r'][z])[mm] for z in range(zmax + dmax)]
    sc = np.array([np.mean([np.corrcoef(G_[z], R_[z + d])[0, 1] for z in range(zmax)]) for d in range(-3, dmax)])
    return int(np.arange(-3, dmax)[np.argmax(sc)]), sc
