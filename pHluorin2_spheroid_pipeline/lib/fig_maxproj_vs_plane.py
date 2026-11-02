
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
mpl.rcParams['svg.fonttype'] = 'none'
def crop(a, c, half=110):
    cy, cx = int(round(c[0])), int(round(c[1])); out = np.full((2*half, 2*half), np.nan, np.float32)
    y0, y1, x0, x1 = max(0, cy-half), min(490, cy+half), max(0, cx-half), min(490, cx+half)
    out[y0-(cy-half):y1-(cy-half), x0-(cx-half):x1-(cx-half)] = a[y0:y1, x0:x1]; return out
picks = []
for c in ['DMSO', 'Baf L', 'Olig M', 'Rot H', '3BP H', 'Oz L', '50uM Ogerin', '20uM OGM']:
    cand = ZQ[(ZQ.condition == c) & (ZQ.timepoint == '72h')]
    if cand.empty: cand = ZQ[(ZQ.condition == c) & (ZQ.timepoint == '48h')]
    r = cand.iloc[(cand.mp_ratio_median - cand.mp_ratio_median.median()).abs().argsort().iloc[0]]
    picks.append((r.timepoint, r.well, c))
cmR = plt.get_cmap('RdBu_r').copy(); cmR.set_bad('black'); cmE = plt.get_cmap('PuOr_r').copy(); cmE.set_bad('black')
fig, axes = plt.subplots(4, len(picks), figsize=(1.75 * len(picks) + 1.3, 7.6), facecolor='black')
rows_ = [('max projection: 488/405', 'mp', 'R', cmR, 0.9, 2.4), ('single equatorial plane: 488/405', 'eq', 'R', cmR, 0.9, 2.4),
         ('max projection: patch map', 'mp', 'res', cmE, -0.25, 0.25), ('single plane: patch map', 'eq', 'res', cmE, -0.25, 0.25)]
for j, (t, w, c) in enumerate(picks):
    k = (t, w)
    for i, (lab, src, key, cm_, lo, hi) in enumerate(rows_):
        mp_ = MPM[k] if src == 'mp' else ZMAPS[k]
        ax = axes[i, j]; ax.set_facecolor('black'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.imshow(crop(mp_[key].astype(np.float32), mp_['centroid']), cmap=cm_, vmin=lo, vmax=hi, interpolation='nearest')
        if i == 0: ax.set_title(f"{LAB.get(c, c)}\n{t} · {w}", color='white', fontsize=6.5, pad=3)
        if j == 0: ax.set_ylabel(lab, color='white', fontsize=6.5)
for i, (lab, src, key, cm_, lo, hi) in enumerate(rows_):
    cax = fig.add_axes([0.925, 0.75 - i * 0.215 + 0.03, 0.01, 0.15])
    cb = fig.colorbar(ScalarMappable(cmap=cm_, norm=Normalize(lo, hi)), cax=cax)
    cb.set_label('488/405\n(higher = more acidic)' if key == 'R' else 'residual\n(orange = acidic)', color='white', fontsize=5.5)
    cb.ax.tick_params(colors='white', labelsize=5); cb.outline.set_edgecolor('white')
fig.suptitle('Max projection vs single chromatic-matched equatorial plane (488 slice z, 405 slice z + 20 µm); 614 × 614 µm crops',
             color='white', fontsize=7.5, y=0.995)
fig.subplots_adjust(left=0.06, right=0.91, top=0.93, bottom=0.01, wspace=0.03, hspace=0.05)
fig.savefig('maxproj_vs_plane_maps.svg', facecolor='black', dpi=200); plt.close(fig)
