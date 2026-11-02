# Spatial overview figure (488/405; higher = more acidic). Needs: feat, dRC, absP, fam, fcol, groups, xdepth, display_order.
def spread(ys, gap):
    o = np.argsort(ys); y = np.array(ys, float)[o]
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i-1] + gap)
    out = np.empty_like(y); out[o] = y; return out

fig = plt.figure(figsize=(14, 10))
gs_ = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
ax = fig.add_subplot(gs_[0, 0]); d72 = dRC[dRC.timepoint=='72h']
lim = [-0.55, 0.8]; ax.plot(lim, lim, color='0.6', lw=0.8, ls='--'); ax.axhline(0, color='0.85', lw=0.6); ax.axvline(0, color='0.85', lw=0.6)
for _, r in d72.iterrows():
    ax.scatter(r.d_core, r.d_rim, s=28, color=fcol[r.family], edgecolor='white', lw=0.4, zorder=3)
for c, off in [('Baf M',(5,2)),('Baf H',(5,-7)),('Rot H',(6,4)),('1uM TMZ',(5,-1)),('Olig M',(-34,0)),
               ('20uM OGM',(5,-2)),('5uM OGM',(5,-2))]:
    r = d72[d72.condition==c].iloc[0]
    ax.annotate(c.replace('uM ','µM '), (r.d_core, r.d_rim), xytext=off, textcoords='offset points', fontsize=5.5, color=fcol[r.family])
ax.text(0.03, 0.97, 'above dashed line = rim shifts\nmore than the centre', transform=ax.transAxes, va='top', fontsize=5.5, color='0.4')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel('Δ centre 488/405 vs DMSO'); ax.set_ylabel('Δ rim 488/405 vs DMSO')
ax.set_title('Rim vs centre shift (72 h, condition means)', fontsize=7, loc='left'); panel_letter(ax, 'a')

for k, tp in enumerate(['48h','72h']):
    ax = fig.add_subplot(gs_[0, 1+k]); ends = []
    for cond, fm in groups:
        P = np.array([absP[(tp, w)] for w in feat[(feat.timepoint==tp)&(feat.condition==cond)]['well']])
        mu, se = np.nanmean(P, 0), np.nanstd(P, 0)/np.sqrt(np.isfinite(P).sum(0))
        ax.plot(xdepth, mu, color=fcol[fm], lw=2 if cond in ('DMSO','Baf H','20uM OGM') else 1.2, ls='--' if cond=='DMSO' else '-')
        ax.fill_between(xdepth, mu-se, mu+se, color=fcol[fm], alpha=0.15, lw=0)
        ends.append((cond, fm, mu[np.isfinite(mu)][-1]))
    ys = spread([e[2] for e in ends], 0.035)
    for (cond, fm, _), y in zip(ends, ys):
        ax.text(318, y, cond.replace('uM ','µM '), fontsize=5.5, color=fcol[fm], va='center')
    ax.axvspan(0, 10, color='0.9', zorder=0); ax.set_xlim(0, 370)
    ax.set_xlabel('distance from spheroid edge (full-res px)'); ax.set_ylabel('488/405 (flat-field corrected)')
    ax.set_title(f'Radial 488/405 profile, {tp} (mean ± SEM, n=3)', fontsize=7, loc='left')
    goodness_arrow(ax, 'higher = more acidic', loc='upper right', axis='y', fontsize=5.5); panel_letter(ax, 'bc'[k])

ax = fig.add_subplot(gs_[1, 0])
for fm, d in feat.groupby('family'):
    ax.scatter(d['ratio_median'], d['resid_sd'], s=12, color=fcol[fm], alpha=0.8, label=fm, edgecolor='none')
o = feat[(feat.condition=='Ace M') & feat.well.isin(['H5','H6'])]
ax.scatter(o['ratio_median'], o['resid_sd'], s=40, facecolor='none', edgecolor='k', lw=0.7)
ax.annotate('Ace M H5/H6 (same wells at every timepoint)', (o['ratio_median'].max(), o['resid_sd'].max()),
            xytext=(8, 0), textcoords='offset points', fontsize=5.5)
ax.set_xlabel('median 488/405 (higher = more acidic)'); ax.set_ylabel('residual SD after removing radial trend')
ax.set_title('Patchiness vs mean (all wells, all timepoints)', fontsize=7, loc='left')
ax.legend(fontsize=5, ncol=2, loc='lower right', frameon=False); panel_letter(ax, 'd')

for k, (m, lab, ttl) in enumerate([('acidic_patch_frac','fraction of spheroid in acidic patches\n(>0.06 above radial trend)','Acidic patch area, 72 h'),
                                     ('corr_length_px','spatial correlation length (full-res px)','Patch size (correlation length), 72 h')]):
    ax = fig.add_subplot(gs_[1, 1+k]); d = feat[feat.timepoint=='72h']
    xs = [c for c in display_order if c in d.condition.unique()]
    for i, c in enumerate(xs):
        v = d[d.condition==c][m].values; fm = fam[c]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=9, color=fcol[fm], zorder=3)
        ax.plot([i-0.32, i+0.32], [np.median(v)]*2, color='k', lw=0.9, zorder=4)
    ax.axhline(d[d.condition=='DMSO'][m].median(), color='0.5', ls=':', lw=0.8)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([c.replace('uM ','µM ') for c in xs], rotation=90, fontsize=4.8)
    ax.set_ylabel(lab); ax.set_title(ttl + ' (black tick = median; dotted = DMSO)', fontsize=7, loc='left'); panel_letter(ax, 'ef'[k])
fig.savefig('spatial_overview.svg', bbox_inches='tight'); plt.close(fig)
