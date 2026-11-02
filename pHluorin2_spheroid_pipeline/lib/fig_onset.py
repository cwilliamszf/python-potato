# Onset of compound effects: 2 h (early; enzyme/transporter timescale) vs 24-72 h, all relative to on-plate DMSO.
# Needs: DT (dose_trend_by_timepoint table), E (effects_all_datasets), CPDS, apply_figure_style, panel_letter.
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
apply_figure_style(sizes=(7, 6, 5.5)); mpl.rcParams['svg.fonttype'] = 'none'
NAME = {'TMZ': 'TMZ', 'Oz': 'Ouabain', 'Baf': 'Bafilomycin A1', 'Rot': 'Rotenone', '3BP': '3-Bromopyruvate', 'Phlor': 'Phloretin',
        'Olig': 'Oligomycin', 'Ace': 'Acetazolamide', 'OGM': 'OGM', 'Ogerin': 'Ogerin'}
TPS4 = ['2h', '24h', '48h', '72h']; DS = [('MCF7_rep', 'MCF7 repeat'), ('HEK293', 'HEK293')]
fig = plt.figure(figsize=(12, 7.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.62, hspace=0.55)

def heat(ax, cols, getter, xl, title, cbl):
    M = np.array([[getter(c, a, b)[0] for a, b in cols] for c in CPDS]); Q = np.array([[getter(c, a, b)[1:] for a, b in cols] for c in CPDS])
    v = 3
    im = ax.imshow(np.clip(M, -v, v), cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            p_, q_ = Q[i, j]; mk = '*' if q_ < 0.1 else ('°' if p_ < 0.05 else '')
            ax.text(j, i, f'{M[i, j]:+.1f}{mk}', ha='center', va='center', fontsize=5, color='white' if abs(M[i, j]) > 2 else '0.15')
    ax.set_yticks(range(len(CPDS))); ax.set_yticklabels([NAME[c] for c in CPDS])
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(xl)
    k = len(cols) // 2; ax.axvline(k - 0.5, color='k', lw=1)
    for j0, (_, lab) in enumerate(DS): ax.text(j0 * k + (k - 1) / 2, -0.9, lab, ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_title(title, pad=18)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label(cbl, fontsize=5.5); cb.ax.tick_params(labelsize=5.5)

def g_time(c, d, t):
    r = DT[(DT.feature == 'ratio_median_cal') & (DT.dataset == d) & (DT.timepoint == t) & (DT.compound == c)].iloc[0]
    return r.slope_per_dmso_sd, r.p, r.q
def g_feat(c, d, f):
    r = DT[(DT.timepoint == '2h') & (DT.dataset == d) & (DT.feature == f) & (DT.compound == c)].iloc[0]
    return r.slope_per_dmso_sd, r.p, r.q

axA = fig.add_subplot(gs[0, 0])
heat(axA, [(d, t) for d, _ in DS for t in TPS4], g_time, [t.replace('h', ' h') for _ in DS for t in TPS4],
     'Mean acidity: dose trend vs on-plate DMSO', 'slope per dose step (DMSO SDs)\nred = more acidic')
panel_letter(axA, 'a')
axC = fig.add_subplot(gs[1, 0])
heat(axC, [(d, f) for d, _ in DS for f in ['sbr488', 'sbr405', 'log_area']], g_feat, ['488', '405', 'area'] * 2,
     'At 2 h: reporter signal (488, 405) and spheroid area', 'slope per dose step (DMSO SDs)')
panel_letter(axC, 'c')

EE = E[E.dataset.isin(['MCF7_exp2_max', 'HEK293_max'])]
CC = {'OGM': '#238B45', 'Phlor': '#C51B7D', 'Baf': '#CB181D', 'Olig': '#2171B5', 'Rot': '#6A51A3', '3BP': '#8C510A'}
hiC = {'OGM': '20uM OGM', 'Phlor': 'Phlor H', 'Baf': 'Baf L', 'Olig': 'Olig H', 'Rot': 'Rot H', '3BP': '3BP M'}
LAB = {'20uM OGM': 'OGM 20 µM', 'Phlor H': 'Phloretin 150 µM', 'Baf L': 'Bafilomycin 5 nM', 'Olig H': 'Oligomycin 300 nM', 'Rot H': 'Rotenone 1 µM', '3BP M': '3BP 50 µM'}
hours = {'2h': 2, '24h': 24, '48h': 48, '72h': 72}
for r_, (grp, ttl) in enumerate([(['OGM', 'Phlor'], 'acute (effect at 2 h)'), (['Baf', 'Olig', 'Rot', '3BP'], 'delayed (no effect at 2 h)')]):
    for c_, (dsn, lab) in enumerate([('MCF7_exp2_max', 'MCF7 repeat'), ('HEK293_max', 'HEK293')]):
        ax = fig.add_subplot(gs[r_, 1 + c_])
        ax.axvspan(-2, 6, color='0.92', zorder=0); ax.axhline(1, color='0.6', lw=0.6, ls=':')
        for cpd in grp:
            g = EE[(EE.dataset == dsn) & (EE.condition == hiC[cpd])].assign(h=lambda x: x.timepoint.map(hours)).sort_values('h')
            ax.plot(g.h, g.ratio_median_cal_fc, marker='o', ms=3, lw=1.3, color=CC[cpd], label=LAB[hiC[cpd]])
            sig = g[g.ratio_median_cal_q < 0.1]; ax.plot(sig.h, sig.ratio_median_cal_fc, 'o', ms=6.5, mfc='none', mec=CC[cpd], mew=0.8)
        ax.set_xticks([2, 24, 48, 72]); ax.set_xlim(-2, 76)
        ax.set_title(f'{lab}: {ttl}', fontsize=7)
        if c_ == 0: ax.set_ylabel('488/405 fold-change vs DMSO\n(higher = more acidic)', fontsize=6)
        if r_ == 1: ax.set_xlabel('hours after treatment (nominal)', fontsize=6)
        ax.legend(fontsize=5, loc='best')
        if c_ == 0: panel_letter(ax, 'b' if r_ == 0 else 'd')
fig.text(0.5, 0.0, '* q < 0.1, ° p < 0.05 (dose trend over DMSO + 3-4 doses, HC3 SEs; BH within dataset x timepoint x feature). Open circles in b, d: that condition vs DMSO q < 0.1. '
         'Grey band = 2 h. Bafilomycin 48-72 h in MCF7 coincides with reporter signal near background (72 h not measurable).', ha='center', fontsize=5.5, color='0.35')
fig.savefig('out/onset_2h_vs_late.svg', bbox_inches='tight'); fig.savefig('out/onset_preview.png', dpi=80, bbox_inches='tight'); plt.close(fig)
