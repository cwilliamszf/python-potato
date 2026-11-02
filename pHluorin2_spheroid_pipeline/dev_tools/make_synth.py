"""Synthetic Cytation C10 export for equivalence testing and benchmarking of the spheroid pipeline.

Layout matches the real export: <base>/<plate folder>/<well>_02_<ch>_1Z<z>_<channel>_001.tif (1960x1960 uint16)
plus a *ZProj* sub-folder with <well>_-2_<ch>_1_ZProj[<channel>]_001.tif.
Disk budget: NT distinct raw stacks are written once and hard-linked into every well; ZProj images are distinct
per well (template max projection x per-well channel gains + noise), so per-well ratios differ.

usage: python make_synth.py exp1 <base>      (28 slices, Bright Field, ZProj, 24 h plate missing 51 ZProj wells)
       python make_synth.py t47d <base> <exp1_base>   (7 slices, Phase Contrast, no ZProj; links exp1 templates)
"""
import os, sys, numpy as np, tifffile
from concurrent.futures import ProcessPoolExecutor

N = 1960; NZ = 28; NT = 12
WELLS = [f'{r}{c}' for r in 'ABCDEFGH' for c in range(1, 13)]
EXP1_PLATES = {'24h': '261001_105300_Plate 1', '48h': '261002_173508_Plate 3', '72h': '261003_211651_Plate 1'}
T47D_PLATES = {'24h': '261019_155302_Plate 1', '48h': '261020_092906_Plate 2', '72h': '261021_092749_Plate 3'}
FLOOR = (6500, 5600)
CH = [(1, 'GFP', 'GFP 469,525'), (2, 'Oxidized roGFP2', 'Oxidized roGFP2 400,525'), (3, 'Bright Field', 'Bright Field')]

def template(t):
    rng = np.random.default_rng(100 + t)
    yy, xx = np.mgrid[:N, :N].astype(np.float32)
    cy, cx = 980 + rng.uniform(-90, 90), 980 + rng.uniform(-90, 90)
    R = rng.uniform(260, 420)
    ang = np.arctan2(yy - cy, xx - cx)
    Rth = R * (1 + 0.04 * np.sin(3 * ang + rng.uniform(0, 6)) + 0.02 * np.sin(5 * ang))
    rr = np.hypot(yy - cy, xx - cx)
    inside = 1 / (1 + np.exp((rr - Rth) / 6))
    vign = 1 - 0.18 * ((yy - 980) ** 2 + (xx - 980) ** 2) / 980 ** 2
    # smooth spatial patches of acidity
    from scipy import ndimage as ndi
    patch = ndi.gaussian_filter(rng.normal(size=(N // 8, N // 8)), 3)
    patch = ndi.zoom(patch / patch.std(), 8, order=1)
    depth = np.clip((Rth - rr) / R, 0, 1)
    slices = {}
    for ci, ch, _ in CH:
        st = np.empty((NZ, N, N), np.uint16)
        for z in range(NZ):
            if ch == 'Bright Field':
                foc = np.exp(-((z - 6) / 5.0) ** 2)
                halo = np.exp(-((rr - Rth) / (10 + 3 * abs(z - 6))) ** 2)
                im = 21000 * vign - 9000 * inside * (0.5 + 0.5 * foc) + 4000 * halo * foc
                im += rng.normal(0, 150, (N, N))
            else:
                zc = 6 if ch == 'GFP' else 8
                foc = np.exp(-((z - zc) / 4.5) ** 2)
                base = FLOOR[ci - 1] + (2200 if ch == 'GFP' else 1300) * vign
                acid = 1 + 0.08 * patch + 0.1 * (1 - depth)
                sig = (3200 if ch == 'GFP' else 2600) * inside * foc * (0.6 + 0.4 * depth)
                sig = sig * (acid if ch == 'GFP' else 1 / np.sqrt(acid))
                im = base + sig + rng.normal(0, 60, (N, N))
            st[z] = np.clip(im, 0, 65000).astype(np.uint16)
        slices[ch] = st
    return slices

def write_template(args):
    t, root = args
    S = template(t)
    d = os.path.join(root, '_templates', f't{t:02d}'); os.makedirs(d, exist_ok=True)
    for ci, ch, _ in CH:
        for z in range(NZ):
            tifffile.imwrite(os.path.join(d, f'{ch}_{z:02d}.tif'), S[ch][z])
        np.save(os.path.join(d, f'max_{ci}.npy'), S[ch].max(0))
        np.save(os.path.join(d, f'min_{ci}.npy'), S[ch].min(0))
    return t

def tmpl_of(tp, w):
    return (WELLS.index(w) * 7 + list(EXP1_PLATES).index(tp)) % NT

def cond_gain(tp, w):
    """per-well gains: column block -> condition effect, plus a few dim (low-signal) and empty wells."""
    rng = np.random.default_rng(WELLS.index(w) * 31 + list(EXP1_PLATES).index(tp))
    row, col = 'ABCDEFGH'.index(w[0]), int(w[1:])
    eff = 1 + 0.05 * np.sin(row * 1.7 + (col - 1) // 3 * 2.3) + 0.02 * list(EXP1_PLATES).index(tp)
    kg = (1 + rng.normal(0, 0.03)) * eff; kr = (1 + rng.normal(0, 0.03)) / eff
    if w in ('C5', 'C6', 'C8', 'C9') and tp != '24h': kg *= 0.05; kr *= 0.05     # Baf M/H late: low signal
    return kg, kr

def write_zproj(args):
    root, tp, folder, w = args
    t = tmpl_of(tp, w); kg, kr = cond_gain(tp, w)
    tdir = os.path.join(root, '_templates', f't{t:02d}')
    zd = os.path.join(root, folder, f'{folder[:13]}_ZProj[Plate]'); os.makedirs(zd, exist_ok=True)
    rng = np.random.default_rng(WELLS.index(w) + 1000 * list(EXP1_PLATES).index(tp))
    for ci, ch, lab in CH:
        mx = np.load(os.path.join(tdir, f'max_{ci}.npy')).astype(np.float32)
        if ch != 'Bright Field':
            fl = FLOOR[ci - 1]; k = kg if ci == 1 else kr
            mx = fl + (mx - fl) * k
            if (tp, w) in (('24h', 'B6'), ('72h', 'D9')): mx = np.full_like(mx, fl + 2000)   # empty field
        elif (tp, w) in (('24h', 'B6'), ('72h', 'D9')):
            mx = np.full_like(mx, 21000)
        mx = mx + rng.normal(0, 20, mx.shape)
        tifffile.imwrite(os.path.join(zd, f'{w}_-2_{ci}_1_ZProj[{lab}]_001.tif'), np.clip(mx, 0, 65000).astype(np.uint16))
    return w

def link_raw(root, folder, tp, w, src_root, nz_keep, bf_name):
    t = tmpl_of(tp, w); tdir = os.path.join(src_root, '_templates', f't{t:02d}')
    pd_ = os.path.join(root, folder); os.makedirs(pd_, exist_ok=True)
    zs = list(range(NZ)) if nz_keep == NZ else [0, 4, 8, 12, 16, 20, 24]
    for ci, ch, _ in CH:
        name = bf_name if ch == 'Bright Field' else ch
        for zi, z in enumerate(zs):
            dst = os.path.join(pd_, f'{w}_02_{ci}_1Z{zi}_{name}_001.tif')
            if not os.path.exists(dst): os.link(os.path.join(tdir, f'{ch}_{z:02d}.tif'), dst)

if __name__ == '__main__':
    mode, root = sys.argv[1], sys.argv[2]
    if mode == 'exp1':
        with ProcessPoolExecutor(4) as ex:
            list(ex.map(write_template, [(t, root) for t in range(NT) if not os.path.exists(os.path.join(root, '_templates', f't{t:02d}', 'min_3.npy'))]))
        print('templates done', flush=True)
        missing24 = {'D10', 'D11', 'D12'} | {f'{r}{c}' for r in 'EFGH' for c in range(1, 13)}
        jobs = [(root, tp, f, w) for tp, f in EXP1_PLATES.items() for w in WELLS if not (tp == '24h' and w in missing24)]
        with ProcessPoolExecutor(4) as ex:
            list(ex.map(write_zproj, jobs, chunksize=8))
        print('zproj done', flush=True)
        for tp, f in EXP1_PLATES.items():
            for w in WELLS: link_raw(root, f, tp, w, root, NZ, 'Bright Field')
        print('raw links done')
    else:
        src = sys.argv[3]
        for tp, f in T47D_PLATES.items():
            for w in WELLS: link_raw(root, f, tp, w, src, 7, 'Phase Contrast')
        print('t47d links done')
