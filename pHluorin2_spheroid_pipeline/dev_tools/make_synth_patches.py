"""Synthetic Cytation plates with KNOWN acid-patch dynamics, for testing step 15 (patch dynamics).

Same file layout as the experiment-1 export (3 plates, 96 wells, 1960 x 1960 px ZProj images; MCF7_exp1 layout).
Every well holds ONE spheroid followed across 24/48/72 h. It grows (radius x1.12 per day, area x1.25), and carries
discrete acidic patches (Gaussian bumps of log 488/405, amplitude 0.28, sigma = s x spheroid radius) whose behaviour
is set by the condition:

  static  (DMSO, Fluorobrite, TMZ, OGM, Ogerin, Baf)  patches fixed in the spheroid frame, scale with the spheroid
  move    (Oz L/M/H)      each patch moves 0.08 / 0.16 / 0.28 spheroid radii per day in its own random direction
  grow    (Rot L/M/H)     patch AREA grows x1.3 / x1.7 / x2.2 per day relative to the spheroid
  emerge  (3BP L/M/H)     3 patches at 24 h; +1 / +2 / +4 new patches per day
  vanish  (Phlor L/M/H)   6 patches at 24 h; -1 / -2 / -3 per day
  rotate  (Olig L/M/H)    nuisance: the whole spheroid turns 20 / 45 / 90 deg per day; patches fixed within it
  merge   (Ace L)         two patches approach and fuse by 72 h (plus 2 static)
  split   (Ace M)         one patch divides into two by 72 h (plus 2 static)   [H5, H6 are excluded wells in config]
  drift   (Ace H)         nuisance: the spheroid moves 150 px per day across the well; patches static
  dim     (1uM TMZ)       static acid patches PLUS 2 signal-loss spots: both channels x0.35 at the centre, no ratio
                          change. With a background fraction < 1 the residual medium makes them look acidic, as on
                          the real plates; step 15's per-patch brightness check should flag them (truth kind='dim').
  puncta_acid   (5uM Ogerin)   40 surface puncta per read in the outer 2-25 um of the projected edge (sigma 0.9 um),
                                 488 signal x(1 + a) at the centre, a ~ U(0.1, 1.0): acidic puncta (ratio up)
  puncta_bright (20uM Ogerin)   the same, but both channels x(1 + a): bright, NOT acidic
  necrotic      (50uM Ogerin)   bright-field core (normalised depth > 0.45) darker by 0 / 20 / 35 % at 24 / 48 / 72 h,
                                 with granular texture; fluorescence unchanged (isolates the bright-field readouts)
  Baf M/H at 48/72 h: signal x0.05 (low-signal QC flag), patches static.
Run the pipeline on these plates with dev_tools/synth_patch_experiment.json (alpha fixed at 0.3, as on the real plates).

A fresh low-amplitude texture (log SD ~0.02) is added at every timepoint, so only the designed patches persist.
Ground truth (patch ids, positions in the spheroid frame, sizes, events) is written to <base>/truth.json.

usage: python make_synth_patches.py <base>       (~6.7 GB; raw z-slices only for the 8 detector-floor wells, as links)
"""
import os, sys, json, numpy as np, tifffile
from concurrent.futures import ProcessPoolExecutor

N = 1960; NZ = 28
WELLS = [f'{r}{c}' for r in 'ABCDEFGH' for c in range(1, 13)]
PLATES = {'24h': '261001_105300_Plate 1', '48h': '261002_173508_Plate 3', '72h': '261003_211651_Plate 1'}
TPS = list(PLATES)
LAYOUT = {'A': ['DMSO', 'Fluorobrite', '1uM TMZ', '0.5uM OGM'], 'B': ['Oz L', 'Oz M', 'Oz H', '1uM OGM'],
          'C': ['Baf L', 'Baf M', 'Baf H', '5uM OGM'], 'D': ['Rot L', 'Rot M', 'Rot H', '20uM OGM'],
          'E': ['3BP L', '3BP M', '3BP H', '0.5uM Ogerin'], 'F': ['Phlor L', 'Phlor M', 'Phlor H', '5uM Ogerin'],
          'G': ['Olig L', 'Olig M', 'Olig H', '20uM Ogerin'], 'H': ['Ace L', 'Ace M', 'Ace H', '50uM Ogerin']}
COND = {f'{r}{c}': LAYOUT[r][(c - 1) // 3] for r in LAYOUT for c in range(1, 13)}
FLOOR = (6500.0, 5600.0); AMP = 0.28
FLOOR_WELLS = ['A2', 'B5', 'C11', 'E2', 'F8', 'G5', 'H2', 'D5']
CH = [(1, 'GFP', 'GFP 469,525'), (2, 'Oxidized roGFP2', 'Oxidized roGFP2 400,525'), (3, 'Bright Field', 'Bright Field')]
LMH = {'L': 0, 'M': 1, 'H': 2}


def phenotype(cond):
    p = cond.split()
    if len(p) == 2 and p[1] in LMH:
        k = LMH[p[1]]
        return {'Oz': ('move', [0.08, 0.16, 0.28][k]), 'Rot': ('grow', [1.3, 1.7, 2.2][k]),
                '3BP': ('emerge', [1, 2, 4][k]), 'Phlor': ('vanish', [1, 2, 3][k]),
                'Olig': ('rotate', [20, 45, 90][k]), 'Ace': (['merge', 'split', 'drift'][k], [None, None, 150][k]),
                'Baf': ('static', None)}[p[0]]
    if cond == '1uM TMZ': return ('dim', None)
    if cond == '5uM Ogerin': return ('puncta_acid', None)
    if cond == '20uM Ogerin': return ('puncta_bright', None)
    if cond == '50uM Ogerin': return ('necrotic', None)
    return ('static', None)


def place(rng, n, s, existing=(), rmax=0.72, sep=3.6, tries=4000):
    """n new patch centres (u, v) in the spheroid frame, >= sep * max(sigma) from each other and from `existing`."""
    out = []
    for _ in range(tries):
        if len(out) == n: break
        r = rmax * np.sqrt(rng.uniform(0.04, 1)); th = rng.uniform(0, 2 * np.pi)
        u, v = r * np.sin(th), r * np.cos(th)
        if all(np.hypot(u - a, v - b) >= sep * max(s, t) for a, b, t in list(existing) + [(x[0], x[1], s) for x in out]):
            out.append((u, v))
    if len(out) < n: raise RuntimeError('could not place patches')
    return out


def design(w):
    """Ground truth for one well: per timepoint the list of patches {id, u, v, s} in the spheroid frame,
    spheroid centre (full-res px), radius, rotation, plus the per-transition events."""
    rng = np.random.default_rng(1000 + WELLS.index(w)); cond = COND[w]; ph, par = phenotype(cond)
    c0 = np.array([980 + rng.uniform(-40, 40), 980 + rng.uniform(-40, 40)]); R0 = rng.uniform(235, 290)
    shape_phase = rng.uniform(0, 2 * np.pi); drift_dir = rng.uniform(0, 2 * np.pi)
    P = {t: [] for t in range(3)}; nid = 0; events = {'48h': {}, '72h': {}}
    def new(u, v, s):
        nonlocal nid; nid += 1; return dict(id=nid, u=float(u), v=float(v), s=float(s))
    if ph in ('static', 'rotate', 'drift', 'dim', 'puncta_acid', 'puncta_bright', 'necrotic'):
        s = 0.11; base = [new(u, v, s) for u, v in place(rng, int(rng.integers(4, 6)) if ph != 'dim' else 3, s)]
        if ph == 'dim':
            base += [dict(new(u, v, s), kind='dim') for u, v in place(rng, 2, s, existing=[(p['u'], p['v'], s) for p in base])]
        for t in range(3): P[t] = [dict(p) for p in base]
    elif ph == 'move':
        s = 0.10; m = par
        for _ in range(5000):     # each patch: random direction; positions valid and separated at every timepoint
            n = 4; th = rng.uniform(0, 2 * np.pi, n); r = 0.72 * np.sqrt(rng.uniform(0.02, 1, n))
            st = np.stack([r * np.sin(th), r * np.cos(th)], 1); dr = rng.uniform(0, 2 * np.pi, n)
            dv = np.stack([np.sin(dr), np.cos(dr)], 1) * m
            pos = [st + t * dv for t in range(3)]
            if all(np.hypot(*p.T).max() < 0.76 for p in pos) and \
               all(min(np.hypot(*(p[i] - p[j])) for i in range(n) for j in range(i)) > 3.6 * s for p in pos): break
        else: raise RuntimeError('move design failed')
        ids = [nid + i + 1 for i in range(n)]; nid += n
        for t in range(3): P[t] = [dict(id=ids[i], u=float(pos[t][i, 0]), v=float(pos[t][i, 1]), s=s) for i in range(n)]
    elif ph == 'grow':
        s0 = 0.075; sfin = s0 * np.sqrt(par ** 2)
        base = place(rng, 3, sfin, rmax=0.62, sep=3.2); ids = [nid + i + 1 for i in range(3)]; nid += 3
        for t in range(3):
            st_ = s0 * np.sqrt(par ** t)
            P[t] = [dict(id=ids[i], u=float(u), v=float(v), s=float(st_)) for i, (u, v) in enumerate(base)]
    elif ph == 'emerge':
        s = 0.085; cur = [new(u, v, s) for u, v in place(rng, 3, s, sep=3.2)]; P[0] = [dict(p) for p in cur]
        for t in (1, 2):
            add = [new(u, v, s) for u, v in place(rng, par, s, existing=[(p['u'], p['v'], p['s']) for p in cur], rmax=0.78, sep=3.2)]
            cur = cur + add; P[t] = [dict(p) for p in cur]; events[TPS[t]]['emerged'] = [p['id'] for p in add]
    elif ph == 'vanish':
        s = 0.085; cur = [new(u, v, s) for u, v in place(rng, 6, s, sep=3.2)]; P[0] = [dict(p) for p in cur]
        for t in (1, 2):
            gone = cur[:par]; cur = cur[par:]; P[t] = [dict(p) for p in cur]
            events[TPS[t]]['vanished'] = [p['id'] for p in gone]
    elif ph in ('merge', 'split'):
        s = 0.085; others = [new(u, v, s) for u, v in place(rng, 2, s, rmax=0.75)]
        oth = np.array([[p['u'], p['v']] for p in others])
        for _ in range(20000):     # pair centre + axis: both end positions inside, all clear of the other patches
            r_, t_ = 0.45 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi); mid = np.array([r_ * np.sin(t_), r_ * np.cos(t_)])
            th = rng.uniform(0, np.pi); e = np.array([np.sin(th), np.cos(th)])
            pts = [mid, mid - 0.24 * e, mid + 0.24 * e]
            if all(np.hypot(*q) < 0.76 for q in pts) and min(np.hypot(*(q - o)) for q in pts for o in oth) > 3.6 * s: break
        else: raise RuntimeError('merge/split design failed')
        a, b = new(0, 0, s), new(0, 0, s); f = new(0, 0, s * 1.35)
        gap = [0.24, 0.16, 0.0] if ph == 'merge' else [0.0, 0.16, 0.24]
        for t in range(3):
            if gap[t] > 0:
                pr = [dict(a, u=float(mid[0] - gap[t] * e[0]), v=float(mid[1] - gap[t] * e[1])),
                      dict(b, u=float(mid[0] + gap[t] * e[0]), v=float(mid[1] + gap[t] * e[1]))]
            else:
                pr = [dict(f, u=float(mid[0]), v=float(mid[1]))]
            P[t] = [dict(p) for p in others] + pr
        if ph == 'merge': events['72h']['merged'] = [[a['id'], b['id'], f['id']]]
        else: events['48h']['split'] = [[f['id'], a['id'], b['id']]]
    tps = {}
    for t, tp in enumerate(TPS):
        c = c0 + (t * par * np.array([np.sin(drift_dir), np.cos(drift_dir)]) if ph == 'drift' else 0)
        tps[tp] = dict(centre=[float(c[0]), float(c[1])], radius=float(R0 * 1.12 ** t),
                       rotation_deg=float(t * par) if ph == 'rotate' else 0.0, shape_phase=float(shape_phase), patches=P[t],
                       necrosis=[0.0, 0.2, 0.35][t] if ph == 'necrotic' else 0.0,
                       puncta=[dict(theta=float(rng.uniform(0, 2 * np.pi)), depth_px=float(rng.uniform(3, 36)),
                                    amp=float(rng.uniform(0.1, 1.0)), kind=ph.split('_')[1]) for _ in range(40)]
                       if ph.startswith('puncta') else [])
    return dict(well=w, condition=cond, phenotype=ph, param=par, timepoints=tps, events=events)


def render(args):
    base, tp, w, T = args
    rng = np.random.default_rng(TPS.index(tp) * 1000 + WELLS.index(w) + 7)
    d = T['timepoints'][tp]; cy, cx = d['centre']; R = d['radius']; rot = np.deg2rad(d['rotation_deg'])
    yy, xx = np.mgrid[:N, :N].astype(np.float32); dy, dx = yy - cy, xx - cx
    rr = np.hypot(dy, dx); ang = np.arctan2(dy, dx) - rot - d['shape_phase']
    Rth = R * (1 + 0.035 * np.sin(3 * ang) + 0.02 * np.sin(5 * ang + 1.0))
    inside = 1 / (1 + np.exp(np.clip((rr - Rth) / 5, -50, 50))); depth = np.clip((Rth - rr) / R, 0, 1)
    vign = 1 - 0.18 * ((yy - 980) ** 2 + (xx - 980) ** 2) / 980 ** 2
    # log-ratio field: designed patches + mild acidic rim + fresh smooth texture
    from scipy import ndimage as ndi
    tex = ndi.zoom(ndi.gaussian_filter(rng.normal(size=(N // 8, N // 8)), 3), 8, order=1)
    L = 0.05 * (1 - depth) + 0.02 * tex / tex.std()
    cr, sr_ = np.cos(rot), np.sin(rot)
    dim = np.ones((N, N), np.float32)            # signal-loss spots: both channels dimmed, ratio unchanged
    for p in d['patches']:
        u, v = p['u'] * cr + p['v'] * sr_, -p['u'] * sr_ + p['v'] * cr     # rotate with the spheroid outline
        py, px, sg = cy + u * R, cx + v * R, p['s'] * R; h = int(4.5 * sg) + 2
        y0, y1, x0, x1 = max(0, int(py) - h), min(N, int(py) + h), max(0, int(px) - h), min(N, int(px) + h)
        bump = np.exp(-((yy[y0:y1, x0:x1] - py) ** 2 + (xx[y0:y1, x0:x1] - px) ** 2) / (2 * sg ** 2))
        if p.get('kind', 'acid') == 'dim': dim[y0:y1, x0:x1] *= 1 - 0.65 * bump
        else: L[y0:y1, x0:x1] += AMP * bump
    gain = 0.05 if (COND[w] in ('Baf M', 'Baf H') and tp != '24h') else 1.0
    pg = np.ones((N, N), np.float32); pr = np.ones((N, N), np.float32)     # surface puncta (full-resolution features)
    for q in d['puncta']:
        a_ = q['theta'] - rot - d['shape_phase']; Re = R * (1 + 0.035 * np.sin(3 * a_) + 0.02 * np.sin(5 * a_ + 1.0))
        py, px = cy + (Re - q['depth_px']) * np.sin(q['theta']), cx + (Re - q['depth_px']) * np.cos(q['theta'])
        y0, y1, x0, x1 = int(py) - 6, int(py) + 7, int(px) - 6, int(px) + 7
        bump = np.exp(-((yy[y0:y1, x0:x1] - py) ** 2 + (xx[y0:y1, x0:x1] - px) ** 2) / (2 * 1.3 ** 2))
        pg[y0:y1, x0:x1] *= 1 + q['amp'] * bump
        if q['kind'] == 'bright': pr[y0:y1, x0:x1] *= 1 + q['amp'] * bump
        q['y'], q['x'] = float(py), float(px)
    Sg = gain * 3000 * inside * (0.6 + 0.4 * depth) * dim * pg; Sr = gain * 2400 * inside * (0.6 + 0.4 * depth) * dim * pr
    g = FLOOR[0] + 2200 * vign + Sg * np.exp(L / 2) + rng.normal(0, 40, (N, N))
    r = FLOOR[1] + 900 * vign + Sr * np.exp(-L / 2) + rng.normal(0, 40, (N, N))
    halo = np.exp(-((rr - Rth) / 12) ** 2)
    b = 21000 * vign - 9000 * inside + 4000 * halo + rng.normal(0, 150, (N, N))
    if d['necrosis'] > 0:          # necrotic core: darker, granular bright field
        core = np.clip((depth - 0.45) / 0.1, 0, 1)
        gran = ndi.gaussian_filter(rng.normal(size=(N, N)).astype(np.float32), 3); gran /= gran.std()
        b -= 12000 * vign * d['necrosis'] * core * (1 + 0.5 * gran)
    folder = PLATES[tp]; zd = os.path.join(base, folder, f'{folder[:13]}_ZProj[Plate]'); os.makedirs(zd, exist_ok=True)
    paths = []
    for (ci, ch, lab), im in zip(CH, (g, r, b)):
        p = os.path.join(zd, f'{w}_-2_{ci}_1_ZProj[{lab}]_001.tif'); paths.append((ci, ch, p))
        tifffile.imwrite(p, np.clip(im, 0, 65000).astype(np.uint16))
    if w in FLOOR_WELLS:          # step 4 estimates the detector floor from raw slices of these wells: link the ZProj
        for ci, ch, p in paths:
            for z in range(NZ):
                dst = os.path.join(base, folder, f'{w}_02_{ci}_1Z{z}_{ch}_001.tif')
                if not os.path.exists(dst): os.link(p, dst)
    return tp, w, d['puncta']


if __name__ == '__main__':
    base = sys.argv[1]; os.makedirs(base, exist_ok=True)
    TRUTH = {w: design(w) for w in WELLS}
    json.dump(TRUTH, open(os.path.join(base, 'truth.json'), 'w'), indent=1)
    jobs = [(base, tp, w, TRUTH[w]) for tp in TPS for w in WELLS]
    with ProcessPoolExecutor(int(os.environ.get('SPHEROID_WORKERS', 4))) as ex:
        for i, (tp, w, pu) in enumerate(ex.map(render, jobs, chunksize=4)):
            TRUTH[w]['timepoints'][tp]['puncta'] = pu          # with image positions (full-res px)
            if (i + 1) % 48 == 0: print(i + 1, '/', len(jobs), flush=True)
    json.dump(TRUTH, open(os.path.join(base, 'truth.json'), 'w'), indent=1)
    print('done')
