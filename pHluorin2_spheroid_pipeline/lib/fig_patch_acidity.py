# Step 15 figure: acidity of the patches themselves, and the per-patch brightness check for signal-loss artefacts.
apply_figure_style(sizes=(7, 6, 5.5))
import matplotlib.pyplot as plt
P = PTALL.merge(SP[['timepoint', 'well', 'qc_exclude', 'compound']], on=['timepoint', 'well'])
P = P[~P.qc_exclude]; S_ = SP[~SP.qc_exclude]
order = [c for c in display_order if c in set(S_.condition)]; xo = {c: k for k, c in enumerate(order)}
pal = plt.get_cmap('tab20')
def col(c): return '0.25' if INFO[c][0] in ('DMSO', 'Fluorobrite') else pal(COMPOUNDS.index(INFO[c][0]) % 20 if INFO[c][0] in COMPOUNDS else 0)
fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.95, bottom=0.12, hspace=0.6, wspace=0.22)
# (a) ratio inside patches vs the rest of the spheroid, per condition (last timepoint)
ax = fig.add_subplot(gs[0, 0:2]); d = S_[S_.timepoint == LAST]
for c, g in d.groupby('condition'):
    ax.scatter(np.full(len(g), xo[c]) - 0.12, g.nonpatch_ratio_median, s=8, color='0.6', lw=0)
    ax.scatter(np.full(len(g), xo[c]) + 0.12, g.patch_ratio_mean, s=8, color=col(c), lw=0)
    for a, b in zip(g.nonpatch_ratio_median, g.patch_ratio_mean): ax.plot([xo[c] - 0.12, xo[c] + 0.12], [a, b], color='0.8', lw=0.4)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
ax.set_ylabel('calibrated 488/405 (higher = more acidic)', fontsize=6)
ax.set_title(f'{LAST}: ratio inside patches (colour, area-weighted) vs non-patch tissue (grey), one pair per spheroid. '
             'Patches are above the rest by construction; compare conditions.', loc='left', fontsize=6.5)
# (b) depth-matched contrast per patch
ax = fig.add_subplot(gs[0, 2]); d = P[P.timepoint == LAST]
for c, g in d.groupby('condition'):
    ax.scatter(np.full(len(g), xo[c]) + np.linspace(-0.25, 0.25, len(g)) if len(g) > 1 else [xo[c]], g.log_contrast_depth,
               s=4, color=col(c), lw=0, alpha=0.7)
    ax.plot([xo[c] - 0.3, xo[c] + 0.3], [g.log_contrast_depth.median()] * 2, color='k', lw=1)
ax.axhline(np.log(1.06), color='tab:orange', ls=':', lw=0.7)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
ax.set_ylabel('ln(patch / same-depth tissue)', fontsize=6); ax.set_title(f'{LAST}: patch contrast at matched depth (per patch; bar = median)', loc='left', fontsize=6.5)
# (c) brightness check
ax = fig.add_subplot(gs[1, 0])
ok = P.sig488_log2_vs_depth.notna()
ax.scatter(P[ok & ~P.signal_loss].sig405_log2_vs_depth, P[ok & ~P.signal_loss].sig488_log2_vs_depth, s=4, color='tab:blue', lw=0, alpha=0.5, label='patch')
ax.scatter(P[ok & P.signal_loss].sig405_log2_vs_depth, P[ok & P.signal_loss].sig488_log2_vs_depth, s=8, color='tab:red', lw=0, label='signal_loss flag')
ax.axvline(SIGNAL_LOSS, color='tab:red', ls=':', lw=0.7); ax.axhline(SIGNAL_LOSS, color='tab:red', ls=':', lw=0.7)
ax.axhline(0, color='0.7', lw=0.5); ax.axvline(0, color='0.7', lw=0.5)
ax.set_xlabel('405 signal, patch vs same-depth tissue (log2)', fontsize=6); ax.set_ylabel('488 signal, patch vs same-depth tissue (log2)', fontsize=6)
ax.set_title('Brightness check: pH patches move the channels apart;\nsignal loss dims both (lower left)', loc='left', fontsize=6.5)
ax.legend(fontsize=5.5, frameon=False)
# (d) fraction of flagged patches per condition
ax = fig.add_subplot(gs[1, 1]); fr = P.groupby('condition').signal_loss.mean().reindex(order)
ax.bar(range(len(order)), fr.values, color=[col(c) for c in order])
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5); ax.set_ylabel('fraction of patches', fontsize=6)
ax.set_title('Patches flagged signal_loss (all timepoints)', loc='left', fontsize=6.5)
# (e) change in acidity of persisting patches
ax = fig.add_subplot(gs[1, 2]); cons = [f'{a}->{b}' for a, b in zip(TPS[:-1], TPS[1:])]
d = TR[(TR.register == 'fixed') & TR.pair.isin(cons)]
for c, g in d.groupby('condition'):
    ax.scatter(np.full(len(g), xo[c]) + np.linspace(-0.25, 0.25, len(g)) if len(g) > 1 else [xo[c]], g.contrast_change_mean, s=5, color=col(c), lw=0)
    ax.plot([xo[c] - 0.3, xo[c] + 0.3], [g.contrast_change_mean.mean()] * 2, color='k', lw=1)
ax.axhline(0, color='0.6', ls=':', lw=0.7)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
ax.set_ylabel('change in depth-matched contrast\nper transition (ln units)', fontsize=6)
ax.set_title('Persisting patches: becoming more (+) or less (-) acidic', loc='left', fontsize=6.5)
fig.savefig(f'patch_acidity{SUF}.svg', bbox_inches='tight'); plt.close(fig)
