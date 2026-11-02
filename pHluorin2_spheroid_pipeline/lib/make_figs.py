# Figures for MCF7 pHluorin2-GPI spheroid screen.
# CONVENTION: pH readout = 488/405 (gfp_mean_raw / rogfp_mean_raw); HIGHER = MORE ACIDIC.
import numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

apply_figure_style(sizes=(7, 6, 5.5))

family_map = {'DMSO':'Control','Fluorobrite':'Control','1uM TMZ':'Control'}
for d, f in [('Oz','Ouabain'),('Baf','Bafilomycin'),('Rot','Rotenone'),('3BP','3-Bromopyruvate'),
             ('Phlor','Phloretin'),('Olig','Oligomycin'),('Ace','Acetazolamide')]:
    for s in 'LMH': family_map[f'{d} {s}'] = f
for c in ['0.5','1','5','20']: family_map[f'{c}uM OGM'] = 'OGM'
for c in ['0.5','5','20','50']: family_map[f'{c}uM Ogerin'] = 'Ogerin'

display_order = ['DMSO','Fluorobrite','1uM TMZ'] + \
    [f'{d} {s}' for d in ['Oz','Baf','Rot','3BP','Phlor','Olig','Ace'] for s in 'LMH'] + \
    ['0.5uM OGM','1uM OGM','5uM OGM','20uM OGM','0.5uM Ogerin','5uM Ogerin','20uM Ogerin','50uM Ogerin']

family_colors = {'Control':'#555555','Bafilomycin':'#D62728','Ouabain':'#FF7F0E','Rotenone':'#9467BD',
    '3-Bromopyruvate':'#8C564B','Phloretin':'#E377C2','Oligomycin':'#1F77B4','Acetazolamide':'#BCBD22',
    'OGM':'#2CA02C','Ogerin':'#17BECF'}
family_bar_colors = {'Control':['#888888','#AAAAAA','#666666'],'Ouabain':['#FFBE69','#FFA726','#E65100'],
    'Bafilomycin':['#EF9A9A','#EF5350','#B71C1C'],'Rotenone':['#CE93D8','#AB47BC','#6A1B9A'],
    '3-Bromopyruvate':['#BCAAA4','#8D6E63','#4E342E'],'Phloretin':['#F48FB1','#E91E63','#880E4F'],
    'Oligomycin':['#90CAF9','#1E88E5','#0D47A1'],'Acetazolamide':['#E6EE9C','#C6D22B','#827717'],
    'OGM':['#A5D6A7','#66BB6A','#2E7D32','#1B5E20'],'Ogerin':['#80DEEA','#26C6DA','#00838F','#004D56']}

meas = pd.read_csv('per_well_measurements.csv')
meas = meas.drop(columns=[c for c in ['ratio_405_488','fc405','dmso_mean','ratio_fc'] if c in meas], errors='ignore')
meas['ratio_488_405'] = meas['gfp_mean_raw'] / meas['rogfp_mean_raw']
meas['drug_family'] = meas['condition'].map(family_map)
dmso = meas[meas['condition']=='DMSO'].groupby('timepoint')['ratio_488_405'].mean()
meas['fc_488_405'] = meas['ratio_488_405'] / meas['timepoint'].map(dmso)
meas.to_csv('per_well_measurements.csv', index=False)

grp = meas.groupby(['timepoint','condition','drug_family']).agg(
    n=('ratio_488_405','count'),
    ratio_mean=('ratio_488_405','mean'), ratio_sem=('ratio_488_405','sem'),
    fc_mean=('fc_488_405','mean'), fc_sem=('fc_488_405','sem'),
    gfp_mean=('gfp_mean_raw','mean'), rogfp_mean=('rogfp_mean_raw','mean'),
    area_mean=('area_px','mean'), area_sem=('area_px','sem')).reset_index()
grp['tp_num'] = grp['timepoint'].map({'24h':24,'48h':48,'72h':72})
grp.to_csv('condition_summary.csv', index=False)
piv = grp.pivot_table(index='condition', columns='timepoint', values='fc_mean')
piv = piv.loc[[c for c in display_order if c in piv.index], ['24h','48h','72h']]

YL = '488/405 fold-change vs DMSO\n(higher = more acidic)'

# ---------- plate heatmaps ----------
rows_ord = list('ABCDEFGH')
mets = [('ratio_488_405','488/405 ratio\n(higher = more acidic)','RdBu_r'),
        ('gfp_mean_raw','488-ex (GFP) mean\nintensity (AU)','Greens'),
        ('area_px','Spheroid area (px²)','viridis')]
cond_of = meas.drop_duplicates('well').set_index('well')['condition'].to_dict()
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ci, (m, ml, cm) in enumerate(mets):
    if m == 'ratio_488_405':
        c0 = meas.loc[meas['condition']=='DMSO', m].mean(); lim = (c0-0.35, c0+0.35)
    else:
        lim = tuple(np.nanpercentile(meas[m], [2, 98]))
    for ti, tp in enumerate(['24h','48h','72h']):
        ax = axes[ti, ci]; grid = np.full((8,12), np.nan)
        for _, r in meas[meas['timepoint']==tp].iterrows():
            grid[rows_ord.index(r['row']), int(r['col'])-1] = r[m]
        im = ax.imshow(grid, cmap=cm, vmin=lim[0], vmax=lim[1], aspect='auto')
        for ri, rl in enumerate(rows_ord):
            for cj in range(12):
                w = f'{rl}{cj+1}'; v = grid[ri, cj]
                txt = 'n.d.' if np.isnan(v) else cond_of.get(w,'').replace('uM','µ').replace(' ','')[:8]
                fr = 0.5 if np.isnan(v) else (v-lim[0])/(lim[1]-lim[0]+1e-9)
                ax.text(cj, ri, txt, ha='center', va='center', fontsize=4,
                        color='white' if (fr < 0.3 or fr > 0.75) else '0.2')
        ax.set_xticks(range(12)); ax.set_xticklabels(range(1,13), fontsize=5)
        ax.set_yticks(range(8)); ax.set_yticklabels(rows_ord, fontsize=5)
        if ti == 0: ax.set_title(ml, fontsize=7, fontweight='bold')
        if ci == 0: ax.set_ylabel(tp, fontsize=7, fontweight='bold', rotation=0, labelpad=25, va='center')
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
fig.suptitle('Plate heatmaps — 488/405 ratio (higher = more acidic), 488-ex intensity, spheroid area', fontsize=9, y=1.01)
plt.tight_layout(); fig.savefig('plate_heatmaps.svg', dpi=130, bbox_inches='tight'); plt.close(fig)

# ---------- condition heatmap ----------
vext = np.nanmax(np.abs(piv.values-1))*1.02
fig, ax = plt.subplots(figsize=(5.5, 10))
im = ax.imshow(piv.values, cmap='RdBu_r', vmin=1-vext, vmax=1+vext, aspect='auto')
for ri in range(piv.shape[0]):
    for cj in range(3):
        v = piv.iloc[ri, cj]
        ax.text(cj, ri, 'n.d.' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                color='white' if (not np.isnan(v) and abs(v-1) > vext*0.6) else '0.2')
ax.set_xticks([0,1,2]); ax.set_xticklabels(['24 h','48 h','72 h'])
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=6.5)
for y in [3,6,9,12,15,18,21,24,28]: ax.axhline(y-0.5, color='white', lw=1.5)
ax.set_title('488/405 fold-change vs DMSO\nred = more acidic, blue = more alkaline', fontsize=8)
cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('Fold-change vs DMSO', fontsize=6)
fig.savefig('condition_heatmap.svg', dpi=130, bbox_inches='tight'); plt.close(fig)

# ---------- time course ----------
dose_markers = {'L':'o','M':'s','H':'^','0.5':'D','1':'o','5':'s','20':'^','50':'P'}
fig, ax = plt.subplots(figsize=(11, 6))
for (cond, fam), g in grp.groupby(['condition','drug_family']):
    g = g.sort_values('tp_num'); ctrl = fam == 'Control'
    key = cond.split('uM')[0] if fam in ('OGM','Ogerin') else cond.split()[-1]
    ax.errorbar(g['tp_num'], g['fc_mean'], yerr=g['fc_sem'], color=family_colors[fam],
                alpha=0.5 if ctrl else 0.85, lw=1 if ctrl else 1.6, ls='--' if ctrl else '-',
                marker=dose_markers.get(key,'o'), ms=4, capsize=2)
ax.axhline(1, color='grey', lw=0.8, ls=':')
for cond in ['20uM OGM','5uM OGM','Baf M','1uM TMZ']:
    r = grp[(grp['condition']==cond)&(grp['timepoint']=='72h')]
    if len(r): ax.text(73.5, r['fc_mean'].values[0], cond, fontsize=5.5, va='center',
                       color=family_colors[r['drug_family'].values[0]])
ax.set_xticks([24,48,72]); ax.set_xticklabels(['24 h','48 h','72 h']); ax.set_xlim(20, 80)
ax.set_xlabel('Time post-treatment'); ax.set_ylabel(YL)
goodness_arrow(ax, 'higher = more acidic', loc='upper right', axis='y', fontsize=5.5)
ax.set_title('OGM ≥5 µM alkalinizes the spheroid surface; most other treatments acidify it (mean ± SEM, n=3)', fontsize=8)
ax.legend(handles=[Line2D([0],[0],color=c,lw=2,label=f) for f,c in family_colors.items()],
          fontsize=5.5, ncol=2, loc='upper left', title='Drug family', title_fontsize=6)
fig.savefig('timecourse_plot.svg', dpi=130, bbox_inches='tight'); plt.close(fig)

# ---------- bar charts ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
for ai, tp in enumerate(['24h','48h','72h']):
    ax = axes[ai]; tg = grp[grp['timepoint']==tp]
    xs, cols, labs, ms, ss = [], [], [], [], []; x = 0; prev = None; famcount = {}
    for cond in display_order:
        r = tg[tg['condition']==cond]
        if r.empty: continue
        r = r.iloc[0]; fam = r['drug_family']
        if prev and fam != prev: x += 0.5
        prev = fam; k = famcount.get(fam, 0); famcount[fam] = k+1
        xs.append(x); cols.append(family_bar_colors[fam][k % len(family_bar_colors[fam])])
        labs.append(cond.replace('uM ','µM\n').replace(' ','\n')); ms.append(r['fc_mean']); ss.append(r['fc_sem']); x += 1
    ax.bar(xs, ms, color=cols, width=0.75, yerr=ss, error_kw=dict(capsize=2, lw=0.8, ecolor='0.3'), zorder=3)
    ax.axhline(1, color='0.4', lw=0.9, ls='--')
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=4.2, rotation=90)
    ax.set_title(tp, fontsize=9, fontweight='bold'); ax.set_ylim(0.55, 1.6)
    if ai == 0: ax.set_ylabel(YL)
    goodness_arrow(ax, 'higher = more acidic', loc='upper left', axis='y', fontsize=5.5)
fig.suptitle('488/405 fold-change vs DMSO at each timepoint (mean ± SEM, n=3)', fontsize=8.5, y=1.01)
plt.tight_layout(); fig.savefig('bar_charts.svg', dpi=130, bbox_inches='tight'); plt.close(fig)

# ---------- dose-response ----------
def hill4(x, bottom, top, ec50, n):
    return bottom + (top-bottom)/(1+(ec50/x)**n)
conc_series = {'OGM':{'0.5uM OGM':0.5,'1uM OGM':1,'5uM OGM':5,'20uM OGM':20},
               'Ogerin':{'0.5uM Ogerin':0.5,'5uM Ogerin':5,'20uM Ogerin':20,'50uM Ogerin':50}}
rank_series = {f: {f'{d} {s}': i+1 for i, s in enumerate('LMH')} for d, f in
               [('Oz','Ouabain'),('Baf','Bafilomycin'),('Rot','Rotenone'),('3BP','3-Bromopyruvate'),
                ('Phlor','Phloretin'),('Olig','Oligomycin'),('Ace','Acetazolamide')]}
tp_colors = {'24h':'#4477AA','48h':'#EE6677','72h':'#228833'}; tp_ls = {'24h':'-','48h':'--','72h':'-.'}
dr = []
fig, axes = plt.subplots(3, 3, figsize=(13, 11)); axes = axes.ravel()
for pi, (drug, cm_) in enumerate(list(conc_series.items()) + list(rank_series.items())):
    ax = axes[pi]; conc = drug in conc_series
    for tp in ['24h','48h','72h']:
        tg = grp[grp['timepoint']==tp]
        pts = [(v, tg[tg['condition']==c].iloc[0]) for c, v in cm_.items() if not tg[tg['condition']==c].empty]
        if len(pts) < 2: continue
        xs = np.array([p[0] for p in pts], float); ys = np.array([p[1]['fc_mean'] for p in pts]); es = np.array([p[1]['fc_sem'] for p in pts])
        ax.errorbar(xs, ys, yerr=es, fmt='o', color=tp_colors[tp], ms=5, capsize=2, elinewidth=0.8, label=tp)
        fitted = False
        if conc and len(xs) >= 4:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    popt, _ = curve_fit(hill4, xs, ys, p0=[1, ys.min(), 5, 2],
                                        bounds=([0.5, 0.2, 0.05, 0.3], [2, 2, 200, 10]), maxfev=20000)
                inr = xs.min() <= popt[2] <= xs.max()
                dr.append(dict(drug=drug, timepoint=tp, bottom_lowdose=popt[0], top_highdose=popt[1],
                               ec50_um=popt[2], hill_n=popt[3], ec50_within_tested_range=inr))
                if inr:
                    xf = np.logspace(np.log10(xs.min()*0.7), np.log10(xs.max()*1.4), 200)
                    ax.plot(xf, hill4(xf, *popt), color=tp_colors[tp], ls=tp_ls[tp], lw=1.4); fitted = True
            except Exception:
                pass
        if not fitted:
            o = np.argsort(xs); ax.plot(xs[o], ys[o], color=tp_colors[tp], ls=tp_ls[tp], lw=1.0, alpha=0.7)
    ax.axhline(1, color='grey', lw=0.7, ls=':')
    if conc: ax.set_xscale('log'); ax.set_xlabel('Concentration (µM)', fontsize=6)
    else: ax.set_xticks([1,2,3]); ax.set_xticklabels(['L','M','H']); ax.set_xlabel('Dose level', fontsize=6)
    if pi % 3 == 0: ax.set_ylabel('488/405 FC vs DMSO\n(higher = more acidic)', fontsize=6)
    ax.set_title(drug, fontsize=8, fontweight='bold'); ax.margins(y=0.1)
    if pi == 0: ax.legend(fontsize=6, title='Timepoint', title_fontsize=6)
for i in range(pi+1, len(axes)): axes[i].set_visible(False)
fig.suptitle('Dose–response of 488/405 (higher = more acidic); 4PL curve drawn only where EC50 falls within tested range', fontsize=8.5, y=1.01)
plt.tight_layout(); fig.savefig('dose_response.svg', dpi=130, bbox_inches='tight'); plt.close(fig)
dr_df = pd.DataFrame(dr); dr_df.to_csv('dose_response_params.csv', index=False)
