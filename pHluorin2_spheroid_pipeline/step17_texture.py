import os, sys, time, gc, json, pickle, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'lib'))
from config import *
from figstyle import apply_figure_style, panel_letter, goodness_arrow
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import scipy.ndimage as ndi, tifffile
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
mpl.rcParams['svg.fonttype'] = 'none'


def LIB(name):
    return os.path.join(HERE, 'lib', name)


def run_lib(name):
    exec(open(LIB(name), encoding='utf-8').read(), globals())


LATE = TPS_ALL[-2:]; LAST = TPS_ALL[-1]; LASTPAIR = f'{TPS_ALL[-2]}->{TPS_ALL[-1]}'   # original: 48h,72h / 72h / 48h->72h
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
T0 = time.time()
display_order = DISPLAY_ORDER_CFG or (['DMSO', 'Fluorobrite', '1uM TMZ'] + [f'{d} {s}' for d in ['Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace'] for s in 'LMH'] + \
    ['0.5uM OGM', '1uM OGM', '5uM OGM', '20uM OGM', '0.5uM Ogerin', '5uM Ogerin', '20uM Ogerin', '50uM Ogerin'])
"""Step 17 - bright-field texture (core darkness, granularity, dark-core fraction, bright field under acid patches)
and full-resolution surface puncta (small acidic or bright spots in the outer 30 um of the projected spheroid, and
cell-scale 488/405 heterogeneity there). Methods and definitions: lib/texture.py. Reads the raw images again at full
resolution (in parallel worker processes, SPHEROID_WORKERS). Needs steps 3 and 4 (and uses step-4 patch maps).
Outputs: texture_features.csv (per spheroid-timepoint), puncta_table.csv (per punctum), bf_maps.npz,
texture_qc.svg"""
run_lib('common.py'); run_lib('stage1_extract.py'); run_lib('stage2_features.py')
from parallel import WellPool
Fv = pd.read_csv('spatial_features_v2.csv'); s1 = pd.read_csv('stage1_status.csv')
cal = json.load(open('calibration.json')); FLO = {k: tuple(v) for k, v in cal['floors'].items()}; ALP = cal['alpha']
MAPS = pickle.load(open('spatial_maps_cal.pkl', 'rb'))
with np.load('stage1_arrays.npz') as Z:
    keys = [tuple(k.split('|')) for k in Z['keys']]; mA = Z['m']
Mk = {k: mA[i] for i, k in enumerate(keys)}
ZPL = dict(zip(zip(s1.timepoint, s1.well), s1.z_plane)) if 'z_plane' in s1 else {}
tasks = []
for tp in TPS:
    zi = zproj_index(PLATES[tp]); ri = None
    for w in WELLS:
        k = (tp, w)
        row = Fv[(Fv.timepoint == tp) & (Fv.well == w)]
        if row.empty: continue                                   # excluded wells / segmentation failures
        zf = zi.get(w)
        if not (zf and all(c in zf for c in 'grb')) or SEGMENTATION == 'v2' or PROJECTION != 'max':
            ri = ri or raw_index(PLATES[tp])
        zp = ZPL.get(k, np.nan); zp = None if zp is None or not np.isfinite(zp) else int(zp)
        tasks.append((tp, w, zf if (zf and SEGMENTATION != 'v2' and PROJECTION == 'max') else None,
                      ri.get(w) if ri else None, zp, Mk[k], (float(row.g_medium.iloc[0]), float(row.r_medium.iloc[0])),
                      FLO[tp], float(ALP[tp]), MAPS[k]['patch'] if k in MAPS else None))
print(f'== {len(tasks)} spheroid-timepoints at full resolution', flush=True)
rows, prow, BF = [], [], {}
with WellPool(HERE, ['common.py', 'stage1_extract.py', 'texture.py'], fallback_ns=globals()) as pool:
    for t, (f, pr, bfm) in zip(tasks, pool.imap('texture_well', tasks)):
        tp, w = t[0], t[1]
        rows.append(dict(timepoint=tp, well=w, condition=COND[w], **f)); BF[(tp, w)] = bfm
        prow += [dict(timepoint=tp, well=w, condition=COND[w], **p) for p in pr]
TX = pd.DataFrame(rows).merge(Fv[['timepoint', 'well', 'qc_exclude']], on=['timepoint', 'well'], how='left')
TX.to_csv('texture_features.csv', index=False); PU = pd.DataFrame(prow); PU.to_csv('puncta_table.csv', index=False)
np.savez_compressed('bf_maps.npz', keys=np.array([f'{t}|{w}' for t, w in BF]), bf=np.stack(list(BF.values())))
print(TX[~TX.qc_exclude.astype(bool)].groupby('timepoint')[['bf_core_vs_rim_log2', 'bf_texture_core', 'bf_glcm_contrast', 'dark_core_frac',
      'bf_patch_vs_depth_log2', 'puncta_per_1000um2', 'frac_acidic_puncta', 'rim_cellscale_sd']].median().round(3).to_string())

# ---- figure: per-condition summaries + example rim crops with detected puncta
run_lib('figstyle.py'); apply_figure_style(sizes=(7, 6, 5.5)); run_lib('texture.py')
Q = TX[~TX.qc_exclude.astype(bool) & (TX.condition != 'Fluorobrite')]
order = [c for c in display_order if c in set(Q.condition)]; xo = {c: i for i, c in enumerate(order)}
INFO = {c: cond_info(c) for c in order}; pal = plt.get_cmap('tab20')
def col(c): return '0.25' if INFO[c][0] == 'DMSO' else pal(COMPOUNDS.index(INFO[c][0]) % 20 if INFO[c][0] in COMPOUNDS else 0) if 'COMPOUNDS' in globals() else 'C0'
run_lib('analysis_stats.py')
fig = plt.figure(figsize=(16, 15)); gs = fig.add_gridspec(4, 4, hspace=0.95, wspace=0.25, left=0.05, right=0.98, top=0.96, bottom=0.05)
panels = [('bf_core_vs_rim_log2', 'Bright field: core vs rim (log2; < 0 = darker core)'), ('bf_texture_core', 'Bright-field core granularity (SD of high-pass)'),
          ('dark_core_frac', 'Dark-core fraction'), ('bf_patch_vs_depth_log2', 'Bright field under acid patches vs same depth (log2)'),
          ('puncta_per_1000um2', 'Surface puncta per 1000 µm² of rim band'), ('frac_acidic_puncta', 'Fraction of puncta that are acidic'),
          ('puncta_contrast_median', 'Punctum 488/405 contrast vs surround (log2, median)'), ('rim_cellscale_sd', 'Cell-scale 488/405 heterogeneity in the rim band')]
for k, (y, ttl) in enumerate(panels):
    ax = fig.add_subplot(gs[k // 2, (k % 2) * 2:(k % 2) * 2 + 2]); off = {t: (i - (len(TPS) - 1) / 2) * 0.22 for i, t in enumerate(TPS)}
    for (c, tp), g in Q.groupby(['condition', 'timepoint']):
        if c not in xo: continue
        ax.scatter(np.full(len(g), xo[c] + off[tp]), g[y], s=5, color=col(c), lw=0, alpha=0.35 + 0.65 * TPS.index(tp) / max(len(TPS) - 1, 1))
    ref = Q[Q.condition == 'DMSO'].groupby('timepoint')[y].mean()
    for tp, v in ref.items(): ax.axhline(v, color='0.6', ls=':', lw=0.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
    ax.set_title(f'{ttl}   (points left -> right within a condition: {", ".join(TPS)}; dotted = DMSO)', loc='left', fontsize=6.5)
fig.savefig('texture_qc.svg', bbox_inches='tight'); plt.close(fig)

# example: rim crops with detected puncta for DMSO and the two conditions with most acidic / most neutral puncta at the last read
last = Q[Q.timepoint == LAST]
cand = [('DMSO', 'DMSO')]
if len(last):
    ac = last.groupby('condition').acidic_puncta_per_1000um2.mean().drop('DMSO', errors='ignore')
    ne = (last.puncta_per_1000um2 - last.acidic_puncta_per_1000um2).groupby(last.condition).mean().drop('DMSO', errors='ignore')
    if len(ac): cand.append((ac.idxmax(), 'most acidic puncta'))
    if len(ne): cand.append((ne.idxmax(), 'most non-acidic puncta'))
fig, axes = plt.subplots(len(cand), 3, figsize=(12, 4 * len(cand)), squeeze=False)
for i, (c, why) in enumerate(cand):
    sub = last[last.condition == c]
    if sub.empty: continue
    w = sub.sort_values('n_puncta').well.iloc[-1]; t = [x for x in tasks if x[0] == LAST and x[1] == w][0]
    g, r, b, _ = load_full(t[2], t[3], t[4], PROJECTION, SEGMENTATION, BF_CHANNEL)
    bg_g = t[7][0] + t[8] * (t[6][0] - t[7][0]); bg_r = t[7][1] + t[8] * (t[6][1] - t[7][1])
    pu = PU[(PU.timepoint == LAST) & (PU.well == w)]
    mF = ndi.zoom(t[5].astype(np.float32), 4, order=1) > 0.5; ys, xs = np.nonzero(mF)
    cy, cx = int(ys.mean()), int(xs.min()) + 90          # left edge of the spheroid
    win = (slice(cy - 110, cy + 110), slice(cx - 110, cx + 110))
    lr = np.log(np.maximum(ndi.gaussian_filter(g - bg_g, 0.8), 1) / np.maximum(ndi.gaussian_filter(r - bg_r, 0.8), 1))
    for j, (img, lab, cm) in enumerate([(g - bg_g, '488 signal', 'gray'), (r - bg_r, '405 signal', 'gray'), (lr, 'ln 488/405', 'magma')]):
        a = img[win]; ax = axes[i, j]; ax.imshow(a, cmap=cm, vmin=np.percentile(a, 2), vmax=np.percentile(a, 99.5))
        ax.contour(mF[win], [0.5], colors='c', linewidths=0.6)
        inw = pu[(pu.y >= win[0].start) & (pu.y < win[0].stop) & (pu.x >= win[1].start) & (pu.x < win[1].stop)]
        for kind, cc in [('acidic', 'tab:red'), ('neutral', 'tab:cyan'), ('alkaline', 'tab:blue')]:
            q = inw[inw.kind == kind]
            ax.scatter(q.x - win[1].start, q.y - win[0].start, s=60, facecolors='none', edgecolors=cc, lw=0.8, label=kind if j == 0 else None)
        ax.set_title(f'{c} ({why}), {w} {LAST}: {lab}', fontsize=7); ax.axis('off')
    axes[i, 0].legend(fontsize=6, loc='lower right')
fig.suptitle('Surface puncta at full resolution (153 x 153 um crops at the spheroid edge; cyan line = mask edge; circles = detected puncta by kind)', fontsize=8)
fig.savefig('puncta_examples.svg', bbox_inches='tight'); plt.close(fig)
print(f'step 17 done in {time.time()-T0:.0f}s')
