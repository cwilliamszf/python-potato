apply_figure_style(sizes=(7, 6, 5.5))
import matplotlib.pyplot as plt
CC = {'DMSO': '0.35', 'Baf L': '#CB181D', 'Olig M': '#2171B5', 'Rot H': '#6A51A3', '3BP H': '#8C510A', 'Oz L': '#E6550D',
      '20uM OGM': '#238B45', '5uM OGM': '#74C476', '50uM Ogerin': '#17BECF', 'Phlor H': '#C51B8A'}
LAB = {'Baf L': 'Bafilomycin 5 nM', 'Olig M': 'Oligomycin 100 nM', 'Rot H': 'Rotenone 1 µM', '3BP H': '3BP 100 µM',
       'Oz L': 'Ouabain 10 nM', '20uM OGM': 'OGM 20 µM', '5uM OGM': 'OGM 5 µM', '50uM Ogerin': 'Ogerin 50 µM',
       'Phlor H': 'Phloretin 150 µM', 'DMSO': 'DMSO'}
TP = '72h'
def spread(ys, gap):
    o = np.argsort(ys); y = np.array(ys, float)[o]
    for i in range(1, len(y)): y[i] = max(y[i], y[i-1] + gap)
    out = np.empty_like(y); out[o] = y; return out
def end_labels(ax, ends, gap, xoff=0.6):
    ys = spread([e[1] for e in ends], gap)
    for (x, y, lab, col), yl in zip(ends, ys):
        ax.annotate(lab, (x, y), xytext=(x + xoff, yl), color=col, fontsize=5.5, va='center', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.4, alpha=0.6))

fig = plt.figure(figsize=(15.5, 9.8)); gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.42)
h_um = np.arange(0, 19) * 10.1

# (a) absolute ratio vs height — global depth artefact
ax = fig.add_subplot(gs[0, 0]); ends = []
for c in ['DMSO', 'Baf L', 'Olig M', '20uM OGM']:
    d = DEP[(DEP.condition == c) & (DEP.region == 'whole') & (DEP.timepoint == TP)]
    st = d.groupby('h_slices').ratio.agg(['mean', 'sem', 'count']); st = st[st['count'] >= 2]
    ax.errorbar(st.index * 10.1, st['mean'], yerr=st['sem'], color=CC[c], lw=1.4, marker='o', ms=2, capsize=1)
    ends.append((st.index[-1] * 10.1, st['mean'].iloc[-1], LAB[c], CC[c]))
end_labels(ax, ends, 0.12, xoff=8); ax.set_xlim(-5, 240)
ax.set_xlabel('focal height above equatorial plane (µm)'); ax.set_ylabel('whole-spheroid 488/405\n(single chromatic-matched plane)')
ax.set_title('Absolute ratio falls with height in every condition', loc='left'); panel_letter(ax, 'a')

# (b) contrast vs DMSO at the same height — depth invariant
ax = fig.add_subplot(gs[0, 1]); ends = []
for c in ['Baf L', 'Oz L', 'Rot H', '3BP H', 'Olig M', 'Phlor H', '50uM Ogerin', '5uM OGM', '20uM OGM']:
    d = DEP[(DEP.condition == c) & (DEP.region == 'whole') & (DEP.timepoint == TP)]
    st = d.groupby('h_slices').log_contrast.agg(['mean', 'count']); st = st[st['count'] >= 2]
    ax.plot(st.index * 10.1, np.exp(st['mean']), color=CC[c], lw=1.4, marker='o', ms=2)
    ends.append((st.index[-1] * 10.1, np.exp(st['mean'].iloc[-1]), LAB[c], CC[c]))
ax.axhline(1, color='0.6', lw=0.7, ls=':'); end_labels(ax, ends, 0.07, xoff=8); ax.set_xlim(-5, 240)
ax.set_xlabel('focal height above equatorial plane (µm)'); ax.set_ylabel('488/405 relative to DMSO\nat the same height')
ax.set_title('Drug contrasts hold at every height', loc='left'); panel_letter(ax, 'b')

# (c) rim − centre vs height
ax = fig.add_subplot(gs[0, 2]); ends = []
for c in ['DMSO', 'Baf L', 'Olig M', 'Oz L', 'Rot H', '3BP H', '20uM OGM']:
    d = rc[(rc.condition == c) & (rc.timepoint == TP)]
    st = d.groupby('h_slices').rim_core.agg(['mean', 'sem', 'count']); st = st[st['count'] >= 2]
    ax.errorbar(st.index * 10.1, st['mean'], yerr=st['sem'], color=CC[c], lw=1.4, marker='o', ms=2, capsize=1)
    ends.append((st.index[-1] * 10.1, st['mean'].iloc[-1], LAB[c], CC[c]))
ax.axhline(0, color='0.6', lw=0.7, ls=':'); end_labels(ax, ends, 0.06, xoff=8); ax.set_xlim(-5, 240)
ax.set_xlabel('focal height above equatorial plane (µm)'); ax.set_ylabel('rim − centre 488/405\n(single plane)')
ax.set_title('DMSO has no acidic rim; drug rims peak at the equator', loc='left'); panel_letter(ax, 'c')

# (d) max-projection source planes
ax = fig.add_subplot(gs[0, 3])
vals = [ZQ.mp_src488_rim, ZQ.mp_src488_core, ZQ.mp_src405_rim, ZQ.mp_src405_core]
labs = ['488 rim', '488 centre', '405 rim', '405 centre']; cols = ['#3A7D44', '#3A7D44', '#6A3D9A', '#6A3D9A']
for i, (v, cl) in enumerate(zip(vals, cols)):
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v * 10.1, s=4, color=cl, alpha=0.5, lw=0)
    ax.plot([i - 0.25, i + 0.25], [np.median(v) * 10.1] * 2, color='k', lw=1.3)
ax.set_xticks(range(4)); ax.set_xticklabels(labs)
ax.set_ylabel('height of slice supplying the\nmax-projection pixel (µm above z0)')
ax.set_title('Max projection pairs 488 from z0 with 405 from far planes', loc='left')
ax.text(0.02, 0.98, f'n = {len(ZQ)} spheroids; median per spheroid', transform=ax.transAxes, va='top', fontsize=5.5)
panel_letter(ax, 'd')

# (e) effect agreement: max-proj vs matched plane
ax = fig.add_subplot(gs[1, 0:2])
fc = {'ratio_median': '#444444', 'rim_minus_core': '#D95F02', 'resid_sd_rob': '#7570B3', 'acid_patch_frac': '#1B9E77',
      'corr_len_rob': '#E7298A', 'patch_boundary_enrich': '#A6761D'}
for f in ZFEAT:
    ax.scatter(E['mp_' + f + '_es'], E['eq_' + f + '_es'], s=9, color=fc[f], alpha=0.75, lw=0,
               label=f"{ZNICE[f]} (ρ = {A1.set_index('feature').loc[f, 'spearman']:.2f})")
lim = [-8, 12]; ax.plot(lim, lim, color='0.6', lw=0.7, ls='--'); ax.axhline(0, color='0.85', lw=0.5); ax.axvline(0, color='0.85', lw=0.5)
ax.set_xlim(lim); ax.set_ylim(-6, 9)
ax.set_xlabel('effect vs same-plate DMSO, max projection (pooled-SD units)')
ax.set_ylabel('effect vs DMSO, single matched\nequatorial plane (pooled-SD units)')
ax.set_title('Spatial drug effects reproduce in a single in-focus plane; small mean shifts do not', loc='left')
ax.legend(fontsize=5.5, frameon=False, loc='upper left', markerscale=1.5)
ax.text(0.99, 0.02, 'one point = condition × timepoint (n = 88); dashed = identity', transform=ax.transAxes, ha='right', fontsize=5.5, color='0.35')
panel_letter(ax, 'e')

# (f) patch-map correspondence and persistence
ax = fig.add_subplot(gs[1, 2])
groups = [('same', PM.r_same, '#08519C'), ('other', PM.r_other, '0.7'),
          ('same', TSz[TSz.pair == '48h->72h'].r_fixed, '#08519C'),
          ('other', TNz[TNz.pair == '48h->72h'].r_fixed, '0.7')]
for i, (lab, v, cl) in enumerate(groups):
    v = v.dropna(); ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=4, color=cl, alpha=0.6, lw=0)
    ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color='k', lw=1.3)
ax.axhline(0, color='0.8', lw=0.6, ls=':'); ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=5.5)
ax.set_ylabel('correlation of detrended patch maps')
ax.text(0.5, -0.13, 'max projection vs single plane', transform=ax.get_xaxis_transform(), ha='center', fontsize=5.5, va='top', clip_on=False)
ax.text(2.5, -0.13, 'single plane, 48 h vs 72 h', transform=ax.get_xaxis_transform(), ha='center', fontsize=5.5, va='top', clip_on=False)
ax.text(0.02, 0.02, 'same = same spheroid; other = random different spheroid', transform=ax.transAxes, fontsize=5.5, color='0.35')
ax.set_title('Patches exist in-focus and persist across days', loc='left'); panel_letter(ax, 'f')

# (g) OGM medium fluorescence in the 405 channel
ax = fig.add_subplot(gs[1, 3])
ogm = MED[MED.condition.str.contains('OGM')].copy(); ogm['dose'] = ogm.condition.str.replace('uM OGM', '').astype(float)
ogr = MED[MED.condition.str.contains('Ogerin')].copy(); ogr['dose'] = ogr.condition.str.replace('uM Ogerin', '').astype(float)
for d_, cl, nm in [(ogm, '#238B45', 'OGM'), (ogr, '#17BECF', 'Ogerin')]:
    for j, tp in enumerate(TPS):
        st = d_[d_.timepoint == tp].groupby('dose').r_excess.mean()
        ax.plot(st.index, st.values, color=cl, lw=1.2, marker='os^'[j], ms=3, alpha=[1, 0.75, 0.5][j],
                label=f'{nm} {tp}')
ax.axhline(1, color='0.6', lw=0.7, ls=':'); ax.set_xscale('log'); ax.set_xticks([0.5, 1, 5, 20, 50]); ax.set_xticklabels(['0.5', '1', '5', '20', '50'])
ax.set_xlabel('compound concentration (µM)'); ax.set_ylabel('405-channel medium signal\n(× position-matched expectation)')
ax.set_title('OGM fluoresces in the 405 channel', loc='left')
ax.legend(fontsize=5, frameon=False, loc='upper left', ncol=2); panel_letter(ax, 'g')
fig.savefig('optical_validation.svg', bbox_inches='tight'); plt.close(fig)
