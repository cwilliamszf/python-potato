# Per-compound figure: rows = vehicle + doses, columns = 3 replicate spheroids x 24/48/72 h (same well = same spheroid).
# Top block: 488/405 map (HIGHER = MORE ACIDIC). Bottom block: residual after radial trend + shared-pattern removal.
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def wells_for(cond):
    return [w for w in WELLS if COND[w] == cond]

def series_for(cpd):
    if SERIES_CFG and cpd in SERIES_CFG:
        return [tuple(x) for x in SERIES_CFG[cpd]]
    if cpd == 'Controls':
        return [('DMSO', 'DMSO'), ('Fluorobrite', 'Fluorobrite'), ('1uM TMZ', 'TMZ 1 µM')]
    if cpd in DOSE_UM:
        return [('DMSO', 'DMSO')] + [(f'{cpd} {s}', f'{s}: {fmt_dose(DOSE_UM[cpd][i])}') for i, s in enumerate('LMH')]
    doses = {'OGM': ['0.5', '1', '5', '20'], 'Ogerin': ['0.5', '5', '20', '50']}[cpd]
    return [('DMSO', 'DMSO')] + [(f'{d}uM {cpd}', f'{d} µM') for d in doses]

def crop(a, c, half=128, fill=np.nan):
    cy, cx = int(round(c[0])), int(round(c[1])); out = np.full((2*half, 2*half), fill, dtype=np.float32)
    y0, y1, x0, x1 = max(0, cy-half), min(490, cy+half), max(0, cx-half), min(490, cx+half)
    out[y0-(cy-half):y1-(cy-half), x0-(cx-half):x1-(cx-half)] = a[y0:y1, x0:x1]; return out

def compound_figure(cpd, MAPS, F2, tag, rlim, reslim=0.15, fname=None):
    ser = series_for(cpd); nr = len(ser)
    cmR = plt.get_cmap('RdBu_r').copy(); cmR.set_bad('black')
    cmE = plt.get_cmap('PuOr_r').copy(); cmE.set_bad('black')
    nT = len(TPS); W = 3 * nT * 1.05 + 0.6; H = 2 * nr * 1.08 + 1.1
    fig = plt.figure(figsize=(W + 1.4, H), facecolor='black')
    wr = sum([[1, 1, 1] + ([0.12] if t < nT - 1 else []) for t in range(nT)], [])
    gs = fig.add_gridspec(2 * nr, len(wr), width_ratios=wr,
                          left=0.09, right=0.9, top=1 - 0.75 / H, bottom=0.25 / H, wspace=0.03, hspace=0.06)
    colpos = [4 * t + j for t in range(nT) for j in range(3)]
    name = {'Controls': 'Controls'}.get(cpd, FULLNAME.get(cpd, cpd))
    for blk, (key, cm_, lo, hi) in enumerate([('R', cmR, rlim[0], rlim[1]), ('lres', cmE, -reslim, reslim)]):
        for i, (cond, lab) in enumerate(ser):
            ws = wells_for(cond)
            for t, tp in enumerate(TPS):
                for j, w in enumerate(ws[:3]):
                    ax = fig.add_subplot(gs[blk * nr + i, colpos[3 * t + j]]); ax.set_facecolor('black')
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values(): sp.set_visible(False)
                    k = (tp, w)
                    if k not in MAPS:
                        ax.text(0.5, 0.5, 'excluded' if w in ('H5', 'H6') and cond == 'Ace M' else 'no mask',
                                color='0.6', ha='center', va='center', fontsize=5, transform=ax.transAxes); continue
                    mp = MAPS[k][tag]
                    ax.imshow(crop(mp[key].astype(np.float32), mp['centroid']), cmap=cm_, vmin=lo, vmax=hi, interpolation='nearest')
                    row = F2[(F2.timepoint == tp) & (F2.well == w)].iloc[0]
                    txt = (f"{row['ratio_median' + tag]:.2f}" if key == 'R' else f"{row['acid_patch_frac' + tag]*100:.0f}%")
                    ax.text(0.03, 0.03, txt, color='white', fontsize=4.5, transform=ax.transAxes, va='bottom')
                    if k in LOWSIG:
                        for sp in ax.spines.values(): sp.set_visible(True); sp.set_edgecolor('yellow'); sp.set_linewidth(0.8)
                    if i == 0 and blk == 0 and True:
                        ax.set_title(f'{tp} · {w}', color='white', fontsize=5.5, pad=2)
                    elif blk == 0 and j == 1 and i > 0:
                        pass
                    if colpos[3 * t + j] == 0:
                        ax.set_ylabel(lab, color='white', fontsize=6, rotation=0, ha='right', va='center', labelpad=4)
                    if blk == 0 and i > 0 and t == 0:
                        pass
                    # replicate well label for non-first rows (small)
                    if i > 0 or blk == 1:
                        ax.text(0.97, 0.97, w, color='0.75', fontsize=4, transform=ax.transAxes, ha='right', va='top')
        # block label + colourbar
        y0 = 1 - (blk + 1) * (1 - 1.0 / H) / 2
        cax = fig.add_axes([0.915, (1 - 0.75 / H) - (blk + 1) * ((1 - 1.0 / H) / 2) + 0.03, 0.012, (1 - 1.0 / H) / 2 - 0.06])
        cb = fig.colorbar(ScalarMappable(cmap=cm_, norm=Normalize(lo, hi)), cax=cax)
        cb.set_label('488/405 (higher = more acidic)' if key == 'R' else 'log residual vs local trend\n(orange = acidic; patch > +6 %)', color='white', fontsize=6)
        cb.ax.tick_params(colors='white', labelsize=5); cb.outline.set_edgecolor('white')
    model = {'': 'no background subtraction', '_fs': 'detector floor subtracted', '_cal': 'calibrated background subtracted'}[tag]
    fig.suptitle(f'{name} — top: 488/405 map (number = spheroid median; higher = more acidic); bottom: patch map after removing the radial trend '
                 f'(number = acidic-patch area)\n{model}; same well = same spheroid across days; 500×500 full-res px crops at identical scale; '
                 f'yellow frame = spheroid not brighter than medium (ratio unreliable)', color='white', fontsize=6.5, y=1 - 0.12 / H)
    if fname: fig.savefig(fname, dpi=200, facecolor='black')
    plt.close(fig)
