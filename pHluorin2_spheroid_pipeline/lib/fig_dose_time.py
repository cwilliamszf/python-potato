# Dose x time grid: rows = features, columns = compounds; one line per dose (single-hue ramp), DMSO grey.
apply_figure_style(sizes=(7, 6, 5.5)); mpl.rcParams['svg.fonttype'] = 'none'
EXd = EX.set_index(['timepoint', 'well'])
gq_ = gq.join(EXd[['edge_excess', 'edge_share']], on=['timepoint', 'well'])
ROWS = [('ratio_median_cal', 'mean acidity\n(median 488/405)'),
        ('resid_sd_rob_cal', 'residual SD after\nradial detrend'),
        ('acid_patch_frac_cal', 'acidic-patch\narea fraction'),
        ('corr_len_rob_cal', 'correlation\nlength (px)'),
        ('edge_share', 'boundary excess acidity\n(outer 20 % − inner 40 %,\nvs DMSO profile)')]
CPDS = FIG_COMPOUNDS_CFG or COMPOUNDS_CFG or ['Oz', 'Baf', 'Rot', '3BP', 'Phlor', 'Olig', 'Ace', 'OGM', 'Ogerin']
HUE = {'Oz': 'Oranges', 'Baf': 'Reds', 'Rot': 'Purples', '3BP': 'copper_r', 'Phlor': 'RdPu', 'Olig': 'Blues',
       'Ace': 'YlOrBr', 'OGM': 'Greens', 'Ogerin': 'GnBu'}
HUE.update(HUE_CFG or {}); HUE = {c: HUE.get(c, 'Greys') for c in CPDS}
x = np.array([float(t.rstrip('h')) for t in TPS])
fig, axes = plt.subplots(len(ROWS), len(CPDS), figsize=(1.72 * len(CPDS), 9.2), sharey='row', squeeze=False)
for j, cpd in enumerate(CPDS):
    d = gq_[gq_.compound.isin([cpd])]; dm = gq_[gq_.compound == 'DMSO']
    ranks = sorted(d.dose_rank.unique()); cm = plt.get_cmap(HUE[cpd])
    cols = {r: cm(0.35 + 0.6 * i / max(1, len(ranks) - 1)) for i, r in enumerate(ranks)}
    for i, (f, lab) in enumerate(ROWS):
        ax = axes[i, j]
        for sub, col, lw, nm in [(dm, '0.55', 1.0, 'DMSO')] + [(d[d.dose_rank == r], cols[r], 1.5, dose_tag(d[d.dose_rank == r])) for r in ranks]:
            st = sub.groupby('timepoint')[f].agg(['mean', 'sem', 'count']).reindex(TPS)
            ax.errorbar(x, st['mean'], yerr=st['sem'], color=col, lw=lw, marker='o', ms=2.5, capsize=1.5, elinewidth=0.6, label=nm)
            for tp_i, tp in enumerate(TPS):
                if st['count'].iloc[tp_i] < 2 and st['count'].iloc[tp_i] >= 1:
                    ax.plot(x[tp_i], st['mean'].iloc[tp_i], 'o', mfc='white', mec=col, ms=3.5, zorder=4)
        if f == 'edge_share': ax.axhline(0, color='0.75', lw=0.6, ls=':')
        ax.set_xticks(x); ax.set_xticklabels([t.rstrip('h') for t in TPS] if i == len(ROWS) - 1 else [])
        ax.margins(x=0.12, y=0.08)
        if j == 0: ax.set_ylabel(lab)
        if i == 0:
            ax.set_title(FULLNAME.get(cpd, cpd), loc='left')
            ax.legend(fontsize=4.8, frameon=False, loc='upper left', handlelength=1.2, borderaxespad=0.2)
        if i == len(ROWS) - 1: ax.set_xlabel('hours')
fig.text(0.006, 0.985, 'higher = more acidic / more heterogeneous', fontsize=5.5, color='0.35', va='top')
fig.tight_layout(rect=(0, 0, 1, 0.98), w_pad=0.3, h_pad=0.6)
fig.savefig('dose_time_spatial_grid.svg'); plt.close(fig)
