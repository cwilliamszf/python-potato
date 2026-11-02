def spread(ys, gap):
    o = np.argsort(ys); y = np.array(ys, float)[o]
    for i in range(1, len(y)): y[i] = max(y[i], y[i-1] + gap)
    out = np.empty_like(y); out[o] = y; return out
apply_figure_style(sizes=(7, 6, 5.5)); mpl.rcParams['svg.fonttype'] = 'none'
CC = {'DMSO': '0.45', 'Oz': '#E6550D', 'Baf': '#CB181D', 'Rot': '#6A51A3', '3BP': '#8C510A', 'Phlor': '#C51B8A',
      'Olig': '#2171B5', 'Ace': '#B8860B', 'OGM': '#238B45', 'Ogerin': '#17BECF', 'TMZ': '0.7'}
PANEL_CPDS = ['DMSO'] + (FIG_COMPOUNDS_CFG or COMPOUNDS)
_pal = plt.get_cmap('tab10').colors
for _i, _c in enumerate(PANEL_CPDS): CC.setdefault(_c, _pal[_i % 10])
fig = plt.figure(figsize=(15, 9.6)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

# (a) tracking: same-spheroid vs different-spheroid pattern correlation
ax = fig.add_subplot(gs[0, 0]); pairs = list(dict.fromkeys(TS.pair))
for i, p in enumerate(pairs):
    a = TS[TS.pair == p].r_fixed.dropna(); b = TN[TN.pair == p].r_fixed.dropna()
    ax.scatter(i - 0.18 + np.random.default_rng(i).uniform(-0.08, 0.08, len(b)), b, s=3, color='0.75', lw=0, zorder=2)
    ax.scatter(i + 0.18 + np.random.default_rng(9 + i).uniform(-0.08, 0.08, len(a)), a, s=6, color='#08519C', lw=0, zorder=3)
    for xx, v in [(i - 0.18, b), (i + 0.18, a)]: ax.plot([xx - 0.13, xx + 0.13], [np.median(v)] * 2, color='k', lw=1.2, zorder=4)
ax.axhline(0, color='0.8', lw=0.6, ls=':')
ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p.replace('->', ' → ') for p in pairs])
ax.set_ylabel('correlation of patch maps\n(normalised spheroid coordinates)')
ax.set_title('Same spheroid keeps its patch layout across days', loc='left')
ax.text(0.02, 0.98, 'blue = same spheroid (n ≈ 75 per pair)\ngrey = different spheroids', transform=ax.transAxes, va='top', fontsize=5.5)
panel_letter(ax, 'a')

# (b) matched-mean deltas by compound (48 + 72 h), acidic-patch fraction and residual SD
ax = fig.add_subplot(gs[0, 1]); order_c = PANEL_CPDS
for k, (h, off, mk) in enumerate([('acid_patch_frac_cal', -0.17, 'o'), ('resid_sd_rob_cal', 0.17, 's')]):
    P_ = MM[h][0]; P_ = P_[P_.timepoint.isin(LATE)]
    sc = P_.delta.abs().median()
    for i, c in enumerate(order_c):
        v = P_[P_.compound == c].delta.values / sc
        ax.scatter(np.full(len(v), i + off) + np.random.default_rng(i + 10 * k).uniform(-0.06, 0.06, len(v)), v, s=5, color=CC[c], marker=mk, alpha=0.55, lw=0)
        ax.plot([i + off - 0.12, i + off + 0.12], [np.mean(v)] * 2, color='k', lw=1.2)
        _q = PF[(PF.compound == c) & (PF.feature == h)].q.values
        q = _q[0] if len(_q) else np.nan
        if q < 0.05: ax.text(i + off, ax.get_ylim()[1] if False else 6.2, '*', ha='center', fontsize=7)
ax.axhline(0, color='0.6', lw=0.7, ls=':'); ax.set_ylim(-6.5, 6.8)
ax.set_xticks(range(len(order_c))); ax.set_xticklabels([FULLNAME.get(c, c) if c not in ('DMSO',) else c for c in order_c], rotation=45, ha='right')
ax.set_ylabel('heterogeneity minus 5 mean-matched spheroids\n(scaled by median |difference|)')
ax.set_title('Heterogeneity at matched mean acidity (48 + 72 h)', loc='left')
ax.text(0.02, 0.02, '● acidic-patch area   ■ residual SD   * q < 0.05 (label permutation)', transform=ax.transAxes, va='bottom', fontsize=5.5)
panel_letter(ax, 'b')

# (c) covariate adjustment: unadjusted vs brightness+size-adjusted effects
ax = fig.add_subplot(gs[0, 2])
for f, mk in [('ratio_median_cal', 'o'), ('acid_patch_frac_cal', 's'), ('resid_sd_rob_cal', '^')]:
    d = ADJ[(ADJ.feature == f) & ADJ.timepoint.isin(LATE)]
    s_ = d.unadj.abs().max()
    ax.scatter(d.unadj / s_, d.adj / s_, s=10, marker=mk, color=[CC.get(cond_info(c)[0], '0.6') for c in d.condition], lw=0, alpha=0.8, label=FEAT[f])
ax.plot([-1.1, 1.1], [-1.1, 1.1], color='0.6', lw=0.7, ls='--'); ax.axhline(0, color='0.85', lw=0.5); ax.axvline(0, color='0.85', lw=0.5)
ax.set_xlabel('effect vs DMSO, unadjusted (scaled)'); ax.set_ylabel('effect vs DMSO, adjusted for\nspheroid brightness and size (scaled)')
ax.set_title('Drug effects survive brightness and size adjustment', loc='left')
ax.legend(fontsize=5, frameon=False, loc='upper left', markerscale=1.2); panel_letter(ax, 'c')

# (d) trajectories in mean x heterogeneity space: highest dose with >=2 usable spheroids at every timepoint
ax = fig.add_subplot(gs[1, 0:2]); ends = []
for c in PANEL_CPDS:
    d = gq[gq.compound == c]
    cnt = d.groupby(['dose_rank', 'timepoint']).size().unstack().reindex(columns=TPS)
    ok_r = [r for r in cnt.index if (cnt.loc[r] >= 2).all()]
    if not ok_r:   # e.g. Bafilomycin when every dose loses signal at a late timepoint: use the dose with most coverage
        ok_r = [cnt.fillna(0).ge(2).sum(1).idxmax()] if len(cnt) else []
    if not ok_r: continue
    r_ = max(ok_r); dd = d[d.dose_rank == r_]
    for w, ws in dd.groupby('well'):
        ws = ws.set_index('timepoint').reindex(TPS)
        ax.plot(ws.ratio_median_cal, ws.resid_sd_rob_cal, color=CC[c], lw=0.4, alpha=0.45)
    mu = dd.groupby('timepoint')[['ratio_median_cal', 'resid_sd_rob_cal']].mean().reindex(TPS)
    ax.plot(mu.ratio_median_cal, mu.resid_sd_rob_cal, color=CC[c], lw=1.8, marker='o', ms=3)
    _p1, _p0 = mu.iloc[-1].values.astype(float), mu.iloc[-2].values.astype(float)
    if np.all(np.isfinite(_p1)) and np.all(np.isfinite(_p0)) and not np.allclose(_p1, _p0):
        ax.annotate('', xy=_p1, xytext=_p0, arrowprops=dict(arrowstyle='->', color=CC[c], lw=1.8))
    lab = 'DMSO' if c == 'DMSO' else f'{FULLNAME.get(c, c)} {dose_tag(dd)}'
    ends.append((mu.ratio_median_cal.iloc[-1], mu.resid_sd_rob_cal.iloc[-1], lab, CC[c]))
ys = spread([e[1] for e in ends], 0.0042) if 'spread' in globals() else [e[1] for e in ends]
xr = max(e[0] for e in ends) + 0.12
for (x0, y0, lab, col), yl in zip(ends, ys):
    ax.annotate(lab, (x0, y0), xytext=(xr, yl), color=col, fontsize=5.5, va='center',
                arrowprops=dict(arrowstyle='-', color=col, lw=0.5, alpha=0.7))
ax.set_xlim(right=xr + 0.45)
ax.set_xlabel('mean acidity (median 488/405; higher = more acidic)'); ax.set_ylabel('residual SD after radial detrend\n(higher = more heterogeneous)')
ax.set_title('Dose–time trajectories: acidification and heterogeneity rise together for Baf, 3BP, Olig, Rot; OGM alkalinizes without heterogeneity' if EXPERIMENT == 'MCF7_exp1' else f'Dose–time trajectories ({EXPERIMENT}): mean acidity vs heterogeneity', loc='left')
ax.text(0.99, 0.02, 'thick = condition mean, ' + ' → '.join(t.rstrip('h') for t in TPS) + ' h (arrow); thin = individual spheroids; highest dose with ≥2 usable spheroids per timepoint',
        transform=ax.transAxes, ha='right', fontsize=5.5, color='0.35')
panel_letter(ax, 'd')

# (e) reproducibility
ax = fig.add_subplot(gs[1, 2])
L_ = LOO[LOO.label == 'condition (compound × dose)'].reset_index(drop=True)
yy = np.arange(len(L_))
ax.barh(yy, L_.accuracy, color='#4A6FA5', height=0.55)
ax.scatter(L_.null_95, yy, marker='|', s=120, color='k', zorder=3)
ax.axvline(L_.chance.iloc[0], color='0.5', ls=':', lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels(L_.features); ax.invert_yaxis()
ax.set_xlabel(f'leave-one-spheroid-out accuracy\n({gq.condition.nunique()} conditions, ' + '/'.join(t.rstrip('h') for t in TPS) + ' h trajectory)')
ax.set_title('Trajectories identify the condition well above chance', loc='left')
ax.text(0.98, 0.02, f'| = 95th pct of label-shuffled null\ndotted = chance (1/{gq.condition.nunique()})', transform=ax.transAxes, ha='right', fontsize=5.5)
for i, v in enumerate(L_.accuracy): ax.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=5.5)
ax.set_xlim(0, max(0.38, float(L_.accuracy.max()) * 1.18)); panel_letter(ax, 'e')
fig.savefig('robustness_tracking_trajectories.svg', bbox_inches='tight'); plt.close(fig)
