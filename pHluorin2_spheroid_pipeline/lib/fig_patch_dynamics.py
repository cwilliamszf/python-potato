# Step 15 figure: patch tracks for DMSO and the highest dose of each compound (first QC-passed well), and per-condition
# summaries of movement, relative growth, emergence / loss, Jaccard distances and the effect of rotation registration.
apply_figure_style(sizes=(7, 6, 5.5))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def _lineages(w, L_fixed):
    """{(tp, label): lineage id} following 'fixed' links of consecutive timepoints; emerged patches get new ids."""
    lin, nxt = {}, [0]
    def new():
        nxt[0] += 1; return nxt[0]
    for t1, t2 in zip(TPS[:-1], TPS[1:]):
        if (t1, w) in PT:
            for lab in PT[(t1, w)].label: lin.setdefault((t1, lab), new())
        d = L_fixed[(L_fixed.well == w) & (L_fixed.pair == f'{t1}->{t2}')]
        for r in d.itertuples():
            if r.event in ('persist', 'split') and r.labelA > 0: lin[(t2, r.labelB)] = lin.get((t1, r.labelA), new())
        for r in d[d.event == 'emerged'].itertuples(): lin[(t2, r.labelB)] = new()
    for tp in TPS:
        if (tp, w) in PT:
            for lab in PT[(tp, w)].label: lin.setdefault((tp, lab), new())
    return lin

def _exemplars():
    rows = []
    for c in ['DMSO'] + [x for x in COMPOUNDS]:
        conds = [k for k, v in INFO.items() if v[0] == c]
        if not conds: continue
        cond = max(conds, key=lambda k: (INFO[k][2], INFO[k][1] if np.isfinite(INFO[k][1]) else 0))
        ws = [w for w in WELLS if COND[w] == cond and all((tp, w) in GOOD and (tp, w) in MAPS for tp in TPS)]
        if ws: rows.append((cond, ws[0]))
    return rows

LF = pd.DataFrame(links); LF = LF[LF.register == 'fixed'] if len(LF) else LF
EXR = _exemplars(); nE = len(EXR); nT = len(TPS)
pal = plt.get_cmap('tab20'); cmE = plt.get_cmap('PuOr_r').copy(); cmE.set_bad('black')
# ---------------------------------------------------------------- figure 1: tracks, exemplars in two columns
nR = (nE + 1) // 2; H = 1.55 * nR + 0.7
fig = plt.figure(figsize=(16, H))
gs = fig.add_gridspec(nR, 2 * (nT + 2), width_ratios=([1] * nT + [1.9, 0.25]) * 2,
                      left=0.05, right=0.99, top=1 - 0.55 / H, bottom=0.1 / H, hspace=0.12, wspace=0.05)
HALF = 120
for i, (cond, w) in enumerate(EXR):
    lin = _lineages(w, LF); row, c0 = i % nR, (i // nR) * (nT + 2)
    for t, tp in enumerate(TPS):
        ax = fig.add_subplot(gs[row, c0 + t]); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('black')
        mp = MAPS[(tp, w)]; cy, cx = mp['centroid']; L = LAB[(tp, w)]
        y0, x0 = int(cy) - HALF, int(cx) - HALF
        ax.imshow(mp['lres'].astype(np.float32), cmap=cmE, vmin=-0.15, vmax=0.15, interpolation='nearest')
        for lab in PT[(tp, w)].label:
            col = pal((lin[(tp, lab)] - 1) % 20)
            ax.contour(L == lab, [0.5], colors=[col], linewidths=0.9)
        if t > 0:
            d = LF[(LF.well == w) & (LF.pair == f'{TPS[t-1]}->{tp}')]
            r = float(mp['eq_r'])
            for e in d[d.event == 'persist'].itertuples():
                ax.annotate('', (cx + r * e.vB, cy + r * e.uB), (cx + r * e.vA, cy + r * e.uA),
                            arrowprops=dict(arrowstyle='->', color='white', lw=0.8))
            for e in d[d.event == 'emerged'].itertuples():
                ax.plot(cx + r * e.vB, cy + r * e.uB, '+', color='white', ms=7, mew=1.2)
        ax.set_xlim(x0, x0 + 2 * HALF); ax.set_ylim(y0 + 2 * HALF, y0)
        n = len(PT[(tp, w)])
        ax.text(0.03, 0.03, f'{n} patches', transform=ax.transAxes, color='white', fontsize=5.5)
        if row == 0: ax.set_title(tp, fontsize=7)
        if t == 0: ax.set_ylabel(f'{cond}\n{w}', fontsize=6.5)
    # per-transition numbers next to the maps
    ax = fig.add_subplot(gs[row, c0 + nT]); ax.axis('off')
    tr = TR[(TR.well == w) & (TR.register == 'fixed')]
    txt = '\n'.join(f'{p}\n  move {r.disp_median:.2f} R  new {r.n_emerged}  lost {r.n_vanished}\n  growth {r.rel_growth_patch_mean:+.2f} log2'
                     if np.isfinite(r.disp_median) else f'{p}\n  new {r.n_emerged}  lost {r.n_vanished}'
                     for p, r in zip(tr.pair, tr.itertuples()))
    ax.text(0.04, 0.5, txt, fontsize=5, va='center', family='monospace')
fig.text(0.05, 1 - 0.22 / H, f'{"Acid" if POL == "acid" else "Alkaline"}-patch tracks: log residual (±0.15), outline colour = same patch lineage, arrow = movement since the '
         'previous read, + = new patch; no rotation registration. DMSO and the highest dose of each compound (first QC-passed well).',
         fontsize=7)
fig.savefig(f'patch_dynamics_tracks{SUF}.svg', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- figure 2: per-condition summaries
fig = plt.figure(figsize=(16, 13.5))
gs = fig.add_gridspec(3, 3, left=0.05, right=0.97, top=0.96, bottom=0.08, hspace=0.75, wspace=0.18)

# ---- summaries (consecutive transitions pooled; one point per spheroid-transition)
cons = [f'{a}->{b}' for a, b in zip(TPS[:-1], TPS[1:])]
order = [c for c in display_order if c in set(TR.condition)]
x = {c: k for k, c in enumerate(order)}
def strip(ax, y, title, reg='fixed', ylab=None, ref=None):
    d = TR[(TR.register == reg) & TR.pair.isin(cons)]
    for c, g in d.groupby('condition'):
        xs = x[c] + np.linspace(-0.25, 0.25, len(g)) if len(g) > 1 else [x[c]]
        col = '0.25' if INFO[c][0] in ('DMSO', 'Fluorobrite') else pal(COMPOUNDS.index(INFO[c][0]) % 20 if INFO[c][0] in COMPOUNDS else 0)
        ax.scatter(xs, g[y], s=5, color=col, alpha=0.7, lw=0)
        ax.plot([x[c] - 0.3, x[c] + 0.3], [g[y].mean()] * 2, color='k', lw=1)
    nl = NU[(NU.register == reg) & NU.pair.isin(cons)][y].median() if y in NU else np.nan
    if np.isfinite(nl): ax.axhline(nl, color='tab:red', ls='--', lw=0.8)
    if ref is not None: ax.axhline(ref, color='0.6', ls=':', lw=0.7)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
    ax.set_title(title, loc='left', fontsize=7); ax.set_ylabel(ylab or y, fontsize=6)
specs = [('disp_per_day', '1  Movement: centroid displacement of linked patches', 'radii per day', None),
         ('jaccard_mask_dist', '1  Patch-mask Jaccard distance (1 = no overlap)', 'Jaccard distance', None),
         ('centroid_jaccard_dist', f'1  Centroid-set Jaccard distance (tolerance {JACC_TOL} R)', 'Jaccard distance', None),
         ('rel_growth_patch_per_day', '2  Patch growth vs spheroid (linked patches)', 'log2 FC per day', 0),
         ('rel_growth_total_per_day', '2  Total patch area relative to spheroid', 'log2 FC per day', 0),
         ('emerged_per_day', '3  New patches', 'per day', None),
         ('vanished_per_day', '3  Lost patches', 'per day', None)]
for k, (y, ttl, yl, ref) in enumerate(specs):
    ax = fig.add_subplot(gs[k // 3, k % 3])
    strip(ax, y, ttl, ylab=yl, ref=ref)
# patch number per timepoint and rotation check
ax = fig.add_subplot(gs[2, 1])
spn = SP[~SP.qc_exclude].groupby(['condition', 'timepoint']).n_patches.mean().unstack()[TPS]
for c in order:
    if c in spn.index: ax.plot(range(nT), spn.loc[c].values, color='0.25' if INFO[c][0] in ('DMSO', 'Fluorobrite') else
                               pal(COMPOUNDS.index(INFO[c][0]) % 20 if INFO[c][0] in COMPOUNDS else 0), lw=0.8, alpha=0.8)
ax.set_xticks(range(nT)); ax.set_xticklabels(TPS); ax.set_ylabel('patches per spheroid (mean)', fontsize=6)
ax.set_title('3  Number of patches over time (one line per condition)', loc='left', fontsize=7)
ax = fig.add_subplot(gs[2, 2])
a = TR[(TR.register == 'fixed') & TR.pair.isin(cons)].set_index(['pair', 'well'])
b = TR[(TR.register == 'rotreg') & TR.pair.isin(cons)].set_index(['pair', 'well'])
j = a[['disp_per_day']].join(b[['disp_per_day', 'rotation_deg']], rsuffix='_rot')
rot = np.abs(((j.rotation_deg + 180) % 360) - 180)
sc = ax.scatter(j.disp_per_day, j.disp_per_day_rot, c=rot, cmap='viridis', s=6, vmin=0, vmax=90)
mx = np.nanmax([j.disp_per_day.max(), j.disp_per_day_rot.max(), 0.1]); ax.plot([0, mx], [0, mx], 'k:', lw=0.6)
fig.colorbar(sc, ax=ax, fraction=0.04).set_label('|rotation found| (deg)', fontsize=5.5)
ax.set_xlabel('displacement, no rotation (R/day)', fontsize=6); ax.set_ylabel('after rotation registration', fontsize=6)
ax.set_title('Rotation check: far below the diagonal =\napparent movement explained by spheroid rotation', loc='left', fontsize=7)
fig.legend([Line2D([], [], color='k'), Line2D([], [], color='tab:red', ls='--'), Line2D([], [], color='0.6', ls=':')],
           ['condition mean', 'chance (random spheroid pairs, median)', 'no change'], loc='lower right', fontsize=6, frameon=False)
fig.savefig(f'patch_dynamics_summary{SUF}.svg', bbox_inches='tight'); plt.close(fig)
