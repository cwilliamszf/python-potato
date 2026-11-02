# ---- archived cell 183
print('cell 183', flush=True)

exec(open('spatial_pass1.py', encoding='utf-8').read())
import time; t0 = time.time()
masks, bgmaps, p1 = run_pass1()
print(f"{time.time()-t0:.0f}s", p1['status'].value_counts().to_dict())

# ---- archived cell 189
print('cell 189', flush=True)

exec(open('spatial_pass2.py', encoding='utf-8').read())
fields, stacks = {}, {}
for tp in PLATES:
    fields[tp], stacks[tp] = build_field(bgmaps, tp)
yy, xx = np.mgrid[:980, :980]; cen = (yy-490)**2 + (xx-490)**2 < 350**2
for tp in PLATES:
    F = fields[tp]; f8 = F[::4, ::4]
    rs = [np.corrcoef(s[np.isfinite(s)], f8[np.isfinite(s)])[0, 1] for s in stacks[tp]]
    print(tp, f"field range in central 700px disc: {F[cen].min():.3f}-{F[cen].max():.3f}",
          f"| median r(well bg vs field)={np.median(rs):.2f}")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, tp in zip(axes, PLATES):
    im = ax.imshow(fields[tp], cmap='RdBu_r', vmin=0.85, vmax=1.15); ax.set_title(f'{tp} ratio flat-field'); ax.axis('off')
fig.colorbar(im, ax=axes, fraction=0.02); fig.savefig('ratio_flatfield.svg', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---- archived cell 191
print('cell 191', flush=True)

ok_keys = set(masks)
def build_field2(tp, min_cov=10):
    st = np.stack([v for (t, w), v in bgmaps.items() if t == tp and (t, w) in ok_keys])
    cov = np.isfinite(st).sum(0); f = np.nanmedian(st, 0); f[cov < min_cov] = np.nan
    valid = np.isfinite(f).astype(float); f0 = np.nan_to_num(f)
    num = ndi.gaussian_filter(f0, (25, 1.5)); den = ndi.gaussian_filter(valid, (25, 1.5))
    filled = np.where(valid > 0, f, num / np.maximum(den, 1e-6))
    filled = ndi.gaussian_filter(filled, 2)
    F = ndi.zoom(filled, 4, order=1)
    return (F / np.median(F[cen])).astype(np.float32), cov
covs = {}
for tp in PLATES:
    fields[tp], covs[tp] = build_field2(tp)
    print(tp, f"central-disc field range {fields[tp][cen].min():.3f}-{fields[tp][cen].max():.3f}; "
              f"grid px with <10 wells: {(covs[tp]<10).mean()*100:.1f}%")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, tp in zip(axes, PLATES):
    im = ax.imshow(fields[tp], cmap='RdBu_r', vmin=0.85, vmax=1.15); ax.set_title(f'{tp} ratio flat-field (inpainted)'); ax.axis('off')
    ax.contour(ndi.zoom((covs[tp] < 10).astype(float), 4, order=0), [0.5], colors='k', linewidths=0.6)
fig.colorbar(im, ax=axes, fraction=0.02); fig.savefig('ratio_flatfield.svg', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---- archived cell 193
print('cell 193', flush=True)

def build_field3(tp, min_cov=10, grow=4):
    st = np.stack([v for (t, w), v in bgmaps.items() if t == tp and (t, w) in ok_keys])
    cov = np.isfinite(st).sum(0); f = np.nanmedian(st, 0)
    hole = ndi.binary_dilation(cov < min_cov, iterations=grow); f[hole] = np.nan
    valid = np.isfinite(f).astype(float)
    num = ndi.gaussian_filter(np.nan_to_num(f), (25, 1.5)); den = ndi.gaussian_filter(valid, (25, 1.5))
    filled = ndi.gaussian_filter(np.where(valid > 0, f, num / np.maximum(den, 1e-6)), 2)
    F = ndi.zoom(filled, 4, order=1)
    return (F / np.median(F[cen])).astype(np.float32), hole
holes = {}
for tp in PLATES:
    fields[tp], holes[tp] = build_field3(tp)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, tp in zip(axes, PLATES):
    im = ax.imshow(fields[tp], cmap='RdBu_r', vmin=0.85, vmax=1.15); ax.axis('off')
    ax.set_title(f'{tp}: 488/405 illumination field', fontsize=8)
    ax.contour(ndi.zoom(holes[tp].astype(float), 4, order=0), [0.5], colors='k', linewidths=0.6)
cb = fig.colorbar(im, ax=axes, fraction=0.02); cb.set_label('relative 488/405 gain')
fig.suptitle('Per-plate 488/405 flat-field from spheroid-free medium (black outline = inpainted region under spheroids)', fontsize=8)
fig.savefig('ratio_flatfield.svg', dpi=130, bbox_inches='tight'); plt.close(fig)

import time; t0 = time.time()
KEEP = {('48h', w) for w in ['A1','A8','C8','D8','E8','G8','D11','H11','F8','B2']}
feat, absP, normP, maps48 = run_pass2(p1, masks, fields, keep_maps=KEEP)
print(f"{time.time()-t0:.0f}s, {len(feat)} wells")

# ---- archived cell 198
print('cell 198', flush=True)

fam = {'DMSO':'Control','Fluorobrite':'Control','1uM TMZ':'TMZ'}
for d, f_ in [('Oz','Ouabain'),('Baf','Bafilomycin'),('Rot','Rotenone'),('3BP','3-Bromopyruvate'),
              ('Phlor','Phloretin'),('Olig','Oligomycin'),('Ace','Acetazolamide')]:
    for s in 'LMH': fam[f'{d} {s}'] = f_
for c in ['0.5','1','5','20']: fam[f'{c}uM OGM'] = 'OGM'
for c in ['0.5','5','20','50']: fam[f'{c}uM Ogerin'] = 'Ogerin'
feat['family'] = feat['condition'].map(fam)
feat.to_csv('spatial_features_per_well.csv', index=False)

pd.set_option('display.width', 200)
# 1. edge-exclusion sensitivity
g = feat.groupby('condition')[['rim_minus_core_e0','rim_minus_core','rim_minus_core_e30']].median()
print("rim - core 488/405 by edge exclusion (0 / 10 / 30 full px), median over all tps:")
print(g.loc[['DMSO','Baf H','3BP H','Olig H','Rot H','20uM OGM','5uM OGM','Phlor M','1uM TMZ']].round(3).to_string())
# 2. channel-specific rim/core
print("\nper-channel rim/core intensity (median):")
print(feat.groupby('family')[['rim_over_core_488','rim_over_core_405']].median().round(3).to_string())
# 3. size dependence within DMSO+Fluorobrite+Phloretin (near-null pH conditions)
null = feat[feat['family'].isin(['Control','Phloretin'])]
r_s, p_s = stats.spearmanr(null['eq_diam_px'], null['rim_minus_core'])
r_all, p_all = stats.spearmanr(feat['eq_diam_px'], feat['rim_minus_core'])
print(f"\nrim-core vs diameter: null-like wells rho={r_s:.2f} (p={p_s:.2g}, n={len(null)}); all wells rho={r_all:.2f} (p={p_all:.2g})")
r_bg = stats.spearmanr(feat.merge(p1[['timepoint','well','bg_ratio_median']])['bg_ratio_median'],
                       feat['rim_minus_core'])
print(f"rim-core vs medium 488/405: rho={r_bg[0]:.2f} p={r_bg[1]:.2g}")

# ---- archived cell 200
print('cell 200', flush=True)

FEATS = ['ratio_median','ratio_cv','ratio_iqr','ratio_skew','rim_ratio','core_ratio','rim_minus_core','radial_slope',
         'transition_depth_px','resid_sd','moran_I_12px','corr_length_px','acidic_patch_frac','alkaline_patch_frac',
         'rim_angular_cv','rim_polarity','acid_centroid_offset','area_px','circularity','solidity','boundary_radius_cv']
def effect_sizes(df):
    out = []
    for tp, d in df.groupby('timepoint'):
        pooled = d.groupby('condition')[FEATS].std().pow(2).mean().pow(0.5)   # pooled within-condition SD
        dm = d[d['condition']=='DMSO'][FEATS].mean()
        cm = d.groupby('condition')[FEATS].mean()
        es = (cm - dm) / pooled; es['timepoint'] = tp; out.append(es.reset_index())
    return pd.concat(out)
ES = effect_sizes(feat)
ES.to_csv('spatial_effect_sizes_vs_DMSO.csv', index=False)
order = [c for c in display_order if c in ES['condition'].unique()]
show = ['ratio_median','rim_minus_core','transition_depth_px','ratio_cv','resid_sd','moran_I_12px','corr_length_px',
        'acidic_patch_frac','rim_angular_cv','area_px','solidity']
e72 = ES[ES.timepoint=='72h'].set_index('condition').loc[order, show]
print("72h effect size (SD units vs DMSO):"); print(e72.round(1).to_string())
print("\nraw medians (all tps) of key spatial metrics:")
print(feat.groupby('condition')[['rim_minus_core','transition_depth_px','resid_sd','moran_I_12px','corr_length_px',
                                  'acidic_patch_frac']].median().loc[order].round(3).to_string())

# ---- archived cell 202
print('cell 202', flush=True)

rho = feat[FEATS].corr(method='spearman')
print("Spearman vs ratio_median:", rho['ratio_median'][['rim_minus_core','resid_sd','moran_I_12px','corr_length_px',
      'acidic_patch_frac','rim_angular_cv','area_px','solidity']].round(2).to_dict())
print("resid_sd vs rim_minus_core:", round(rho.loc['resid_sd','rim_minus_core'], 2))
# heterogeneity relative to mean
feat['resid_cv'] = feat['resid_sd'] / feat['ratio_median']
print("\nAce M wells:"); print(feat[feat.condition=='Ace M'][['timepoint','well','ratio_median','resid_sd','acidic_patch_frac','solidity']].round(3).to_string())
# where does the mean shift come from? delta rim vs delta core relative to same-plate DMSO (72h)
rows = []
for tp, d in feat.groupby('timepoint'):
    dm = d[d.condition=='DMSO'][['rim_ratio','core_ratio']].mean()
    cm = d.groupby('condition')[['rim_ratio','core_ratio']].mean()
    x = (cm - dm); x['timepoint'] = tp; rows.append(x.reset_index())
dRC = pd.concat(rows); dRC.columns = ['condition','d_rim','d_core','timepoint']
print("\n72h shift vs DMSO (488/405 units):")
print(dRC[dRC.timepoint=='72h'].set_index('condition').loc[['Baf L','Baf M','Baf H','Olig M','Rot H','3BP H','1uM TMZ',
      'Oz L','5uM OGM','20uM OGM','20uM Ogerin','Phlor M']][['d_rim','d_core']].round(3).to_string())

# ---- archived cell 204
print('cell 204', flush=True)

apply_figure_style(sizes=(7, 6, 5.5))
fcol = {'Control':'#555555','TMZ':'#999999','Bafilomycin':'#D62728','Ouabain':'#FF7F0E','Rotenone':'#9467BD',
        '3-Bromopyruvate':'#8C564B','Phloretin':'#E377C2','Oligomycin':'#1F77B4','Acetazolamide':'#BCBD22',
        'OGM':'#2CA02C','Ogerin':'#17BECF'}
feat['family'] = feat['condition'].map(fam)
dRC['family'] = dRC['condition'].map(fam)
xdepth = ABS_BINS*2 + 10   # bin centres, full-res px from edge

fig = plt.figure(figsize=(14, 9.5))
gs_ = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)
# (a) delta rim vs delta core, 72h
ax = fig.add_subplot(gs_[0, 0]); d72 = dRC[dRC.timepoint=='72h']
lim = [-0.55, 0.8]; ax.plot(lim, lim, color='0.6', lw=0.8, ls='--'); ax.axhline(0, color='0.85', lw=0.6); ax.axvline(0, color='0.85', lw=0.6)
for _, r in d72.iterrows():
    ax.scatter(r.d_core, r.d_rim, s=28, color=fcol[r.family], edgecolor='white', lw=0.4, zorder=3)
for c in ['Baf M','Baf H','Rot H','1uM TMZ','Olig M','20uM OGM','5uM OGM','Oz L']:
    r = d72[d72.condition==c].iloc[0]; ax.annotate(c, (r.d_core, r.d_rim), xytext=(4, -2), textcoords='offset points', fontsize=5.5, color=fcol[r.family])
ax.text(0.03, 0.97, 'above line: rim shifts more\nthan the centre', transform=ax.transAxes, va='top', fontsize=5.5, color='0.4')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel('Δ centre 488/405 vs DMSO'); ax.set_ylabel('Δ rim 488/405 vs DMSO')
ax.set_title('Acidification concentrates at the rim; OGM shifts rim and centre equally (72 h)', fontsize=7, loc='left')
panel_letter(ax, 'a')

# (b,c) radial profiles 48h / 72h
groups = [('DMSO','Control'),('Baf H','Bafilomycin'),('Olig M','Oligomycin'),('Rot H','Rotenone'),
          ('3BP H','3-Bromopyruvate'),('20uM OGM','OGM'),('20uM Ogerin','Ogerin')]
for k, tp in enumerate(['48h','72h']):
    ax = fig.add_subplot(gs_[0, 1+k])
    for cond, fm in groups:
        P = np.array([absP[(tp, w)] for w in feat[(feat.timepoint==tp)&(feat.condition==cond)]['well']])
        mu, se = np.nanmean(P, 0), np.nanstd(P, 0)/np.sqrt(np.isfinite(P).sum(0))
        lw = 2 if cond in ('DMSO','Baf H','20uM OGM') else 1.2
        ax.plot(xdepth, mu, color=fcol[fm], lw=lw, ls='--' if cond=='DMSO' else '-')
        ax.fill_between(xdepth, mu-se, mu+se, color=fcol[fm], alpha=0.15, lw=0)
        ax.text(xdepth[-1]+6, mu[np.isfinite(mu)][-1], cond, fontsize=5.5, color=fcol[fm], va='center')
    ax.axvspan(0, 10, color='0.9', zorder=0); ax.set_xlim(0, 360)
    ax.set_xlabel('distance from spheroid edge (full-res px)'); ax.set_ylabel('488/405 (flat-field corrected)')
    ax.set_title(f'{tp}: acidic shell ~60–100 px deep; OGM profile flat', fontsize=7, loc='left')
    goodness_arrow(ax, 'higher = more acidic', loc='upper right', axis='y', fontsize=5.5)
    panel_letter(ax, 'bc'[k])

# (d) heterogeneity vs mean, all wells
ax = fig.add_subplot(gs_[1, 0])
for fm, d in feat.groupby('family'):
    ax.scatter(d['ratio_median'], d['resid_sd'], s=12, color=fcol[fm], alpha=0.8, label=fm, edgecolor='none')
o = feat[(feat.condition=='Ace M') & feat.well.isin(['H5','H6'])]
ax.scatter(o['ratio_median'], o['resid_sd'], s=40, facecolor='none', edgecolor='k', lw=0.7)
ax.annotate('Ace M wells H5/H6\n(well-specific, all timepoints)', (o['ratio_median'].iloc[2], o['resid_sd'].iloc[2]),
            xytext=(8, -4), textcoords='offset points', fontsize=5.5)
ax.set_xlabel('median 488/405 (higher = more acidic)'); ax.set_ylabel('residual SD after removing radial trend')
ax.set_title('Patchiness rises with acidification; 20 µM OGM spheroids are the most uniform', fontsize=7, loc='left')
ax.legend(fontsize=5, ncol=2, loc='upper left', frameon=False, markerscale=1.2)
panel_letter(ax, 'd')

# (e) acidic patch fraction and (f) correlation length by condition at 72h (strip + median)
for k, (m, lab, ttl) in enumerate([('acidic_patch_frac','fraction of spheroid in acidic patches\n(>0.06 above radial trend)','Acidic patches expand with Baf, Olig, Rot H, 3BP; vanish with OGM'),
                                     ('corr_length_px','spatial correlation length (full-res px)','Baf and 3BP make patches larger (longer correlation length)')]):
    ax = fig.add_subplot(gs_[1, 1+k]); d = feat[feat.timepoint=='72h']
    xs = [c for c in display_order if c in d.condition.unique()]
    for i, c in enumerate(xs):
        v = d[d.condition==c][m].values; fm = fam[c]
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=9, color=fcol[fm], zorder=3)
        ax.plot([i-0.3, i+0.3], [np.median(v)]*2, color=fcol[fm], lw=1.4)
    ax.axhline(d[d.condition=='DMSO'][m].median(), color='0.5', ls=':', lw=0.8)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([c.replace('uM ','µM ') for c in xs], rotation=90, fontsize=4.8)
    ax.set_ylabel(lab); ax.set_title(f'72 h: {ttl}', fontsize=7, loc='left'); panel_letter(ax, 'ef'[k])
fig.savefig('spatial_overview.svg', dpi=150, bbox_inches='tight'); plt.close(fig)

# ---- archived cell 208
print('cell 208', flush=True)

import inspect
src_cell = '''
'''
plt.close('all')
_orig = goodness_arrow
def goodness_arrow(ax, text, loc='upper right', axis='y', fontsize=None):
    return _orig(ax, text, loc='upper left' if loc=='upper center' else loc, axis=axis, fontsize=fontsize)
exec(open('spatial_overview_fig.py', encoding='utf-8').read()) if os.path.exists('spatial_overview_fig.py') else None

# ---- archived cell 210
print('cell 210', flush=True)

goodness_arrow = _orig
exec(open('spatial_overview_fig.py', encoding='utf-8').read())

# ---- archived cell 212
print('cell 212', flush=True)

exec(open('spatial_overview_fig.py', encoding='utf-8').read())

# ---- archived cell 214
print('cell 214', flush=True)

s = open('spatial_overview_fig.py', encoding='utf-8').read().replace(
    "ax.legend(fontsize=5, ncol=2, loc='upper right', frameon=False)", "ax.legend(fontsize=5, ncol=2, loc='lower right', frameon=False)")
open('spatial_overview_fig.py', 'w', encoding='utf-8').write(s); exec(s)

# ---- maps: ratio + detrended residual (48 h)
zi48 = zproj_index(PLATES['48h'])
g_, r_, b_ = load_well('48h', 'H6', zi48, None)
_, _, _, Rh6 = well_features(g_, r_, masks[('48h','H6')], fields['48h'])
maps48[('48h','H6')] = np.where(masks[('48h','H6')], Rh6, np.nan)
sel = [('A1','DMSO'),('C8','Baf H'),('G8','Olig H'),('E8','3BP H'),('D8','Rot H'),('D11','20 µM OGM'),('H6','Ace M (well H6)')]
def resid_map(R, m):
    d = ndi.distance_transform_edt(m); nd = d / d.max()
    fine = np.linspace(0, 1, 21); c_ = (fine[:-1]+fine[1:])/2
    tr = np.array([np.nanmedian(R[(nd > a) & (nd <= b)]) for a, b in zip(fine[:-1], fine[1:])])
    res = np.where(m, R - np.interp(nd, c_, tr), np.nan)
    sm = ndi.gaussian_filter(np.nan_to_num(res), 3) / np.maximum(ndi.gaussian_filter(m.astype(float), 3), 1e-3)
    return np.where(m & (d > EDGE), sm, np.nan)
HALF = 210
cm1 = plt.get_cmap('RdBu_r').copy(); cm1.set_bad('black'); cm2 = plt.get_cmap('PuOr_r').copy(); cm2.set_bad('black')
fig, axes = plt.subplots(2, len(sel), figsize=(2.1*len(sel)+1, 4.8), facecolor='black')
for j, (w, lab) in enumerate(sel):
    m = masks[('48h', w)]; R = maps48[('48h', w)]
    cy, cx = map(int, measure.regionprops(m.astype(np.uint8))[0].centroid)
    def crop(a):
        c = np.full((2*HALF, 2*HALF), np.nan); y0, y1, x0, x1 = max(0, cy-HALF), min(980, cy+HALF), max(0, cx-HALF), min(980, cx+HALF)
        c[y0-(cy-HALF):y1-(cy-HALF), x0-(cx-HALF):x1-(cx-HALF)] = a[y0:y1, x0:x1]; return c
    f_ = feat[(feat.timepoint=='48h') & (feat.well==w)].iloc[0]
    axes[0, j].imshow(crop(R), cmap=cm1, vmin=0.8, vmax=2.0)
    axes[0, j].set_title(lab, color='white', fontsize=8)
    axes[1, j].imshow(crop(resid_map(R, m)), cmap=cm2, vmin=-0.15, vmax=0.15)
    axes[1, j].set_title(f"patch {f_.acidic_patch_frac*100:.0f}% · ξ={f_.corr_length_px:.0f}px", color='white', fontsize=6.5)
    for a in axes[:, j]: a.axis('off'); a.set_facecolor('black')
fig.subplots_adjust(left=0.01, right=0.9, wspace=0.03, hspace=0.12, top=0.86)
for row, (cm_, lo, hi, lab) in enumerate([(cm1, 0.8, 2.0, '488/405\n(higher = more acidic)'), (cm2, -0.15, 0.15, 'residual after\nradial trend')]):
    cax = fig.add_axes([0.915, 0.5 - row*0.43 + 0.03, 0.01, 0.33])
    cb = fig.colorbar(ScalarMappable(cmap=cm_, norm=Normalize(lo, hi)), cax=cax)
    cb.set_label(lab, color='white', fontsize=7); cb.ax.tick_params(colors='white', labelsize=6)
fig.suptitle('48 h: flat-field-corrected 488/405 maps (top) and local patchiness after removing the radial gradient (bottom; orange = more acidic than expected for that depth)',
             color='white', fontsize=8)
fig.savefig('spatial_maps_48h.svg', dpi=170, facecolor='black'); plt.close(fig)

# ---- effect-size heatmap
show = ['ratio_median','rim_ratio','core_ratio','rim_minus_core','ratio_cv','resid_sd','acidic_patch_frac',
        'alkaline_patch_frac','corr_length_px','moran_I_12px','rim_angular_cv','area_px','solidity','boundary_radius_cv']
nice = {'ratio_median':'median 488/405','rim_ratio':'rim 488/405','core_ratio':'centre 488/405','rim_minus_core':'rim − centre',
        'ratio_cv':'CV','resid_sd':'residual SD','acidic_patch_frac':'acidic patch area','alkaline_patch_frac':'alkaline patch area',
        'corr_length_px':'correlation length','moran_I_12px':"Moran's I",'rim_angular_cv':'rim angular CV',
        'area_px':'area','solidity':'solidity','boundary_radius_cv':'edge roughness'}
M = np.full((len(order), 3*len(show)), np.nan)
for i, c in enumerate(order):
    for k, ft in enumerate(show):
        for t, tp in enumerate(['24h','48h','72h']):
            v = ES[(ES.condition==c) & (ES.timepoint==tp)][ft]
            if len(v): M[i, 3*k+t] = v.values[0]
fig, ax = plt.subplots(figsize=(13, 9))
im = ax.imshow(np.clip(M, -5, 5), cmap='RdBu_r', vmin=-5, vmax=5, aspect='auto')
for k in range(1, len(show)): ax.axvline(3*k - 0.5, color='white', lw=2)
for y in [3,6,9,12,15,18,21,24,28]: ax.axhline(y-0.5, color='white', lw=1)
ax.set_xticks([3*k+1 for k in range(len(show))]); ax.set_xticklabels([nice[s] for s in show], rotation=40, ha='right')
ax.set_yticks(range(len(order))); ax.set_yticklabels([c.replace('uM ','µM ') for c in order], fontsize=6)
ax.set_xticks([3*k+t for k in range(len(show)) for t in range(3)], minor=True)
ax.tick_params(axis='x', which='minor', length=2)
ax.set_title('Spatial feature effect sizes vs same-plate DMSO (each block = 24 / 48 / 72 h; units = pooled replicate SD; red = higher)', fontsize=8, loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.02); cb.set_label('effect size (SD units, clipped ±5)')
fig.savefig('spatial_feature_heatmap.svg', dpi=150, bbox_inches='tight'); plt.close(fig)

# ---- archived cell 216
print('cell 216', flush=True)

def well_R(tp, w, zi, ri=None):
    g, r, b = load_well(tp, w, zi, ri)
    gs, rs = ndi.gaussian_filter(g, 1), ndi.gaussian_filter(r, 1)
    return gs / np.maximum(rs, 1) / fields[tp]
# polarity phase from 16 rim sectors, using the already-computed maps where possible is limited; recompute for 48 h & 72 h
from collections import defaultdict
resid_stack = defaultdict(list); phases = defaultdict(list)
for tp in ['48h','72h']:
    zi = zproj_index(PLATES[tp])
    for w in WELLS:
        if (tp, w) not in masks: continue
        m = masks[(tp, w)]; R = well_R(tp, w, zi)
        res = resid_map(np.where(m, R, np.nan), m)
        resid_stack[tp].append(res.astype(np.float32))
        d = ndi.distance_transform_edt(m); p = measure.regionprops(m.astype(np.uint8))[0]
        yy, xx = np.indices(m.shape); ang = np.arctan2(yy - p.centroid[0], xx - p.centroid[1])
        band = (d > EDGE) & (d <= EDGE + RIMW)
        sec = ((ang[band] + np.pi) / (2*np.pi) * 16).astype(int) % 16
        sm = np.array([np.median(R[band][sec == k]) for k in range(16)])
        phases[tp].append(np.angle(np.fft.fft(sm - sm.mean())[1]))
for tp in ['48h','72h']:
    ph = np.array(phases[tp]); Rbar = np.abs(np.mean(np.exp(1j*ph)))
    n = len(ph); z = n * Rbar**2; p_ray = np.exp(-z) * (1 + (2*z - z**2)/(4*n))
    S = np.stack(resid_stack[tp]); mean_res = np.nanmean(S, 0)
    rs_ = [np.corrcoef(s[np.isfinite(s) & np.isfinite(mean_res)], mean_res[np.isfinite(s) & np.isfinite(mean_res)])[0, 1] for s in S]
    print(f"{tp}: rim-polarity phase concentration R={Rbar:.2f} (Rayleigh p={p_ray:.1e}, n={n}); "
          f"median r(well residual vs plate-mean residual)={np.median(rs_):.2f}")

# ---- archived cell 219
print('cell 219', flush=True)

for tp in ['48h','72h']:
    ph = np.array(phases[tp]); ph = ph[np.isfinite(ph)]
    Rbar = np.abs(np.mean(np.exp(1j*ph))); n = len(ph); z = n*Rbar**2
    print(f"{tp}: rim-polarity direction concentration R={Rbar:.2f}, Rayleigh p={np.exp(-z)*(1+(2*z-z**2)/(4*n)):.1e}, "
          f"mean direction {np.degrees(np.angle(np.mean(np.exp(1j*ph)))):.0f} deg (n={n})")
keys = {tp: [(tp, w) for w in WELLS if (tp, w) in masks] for tp in ['48h','72h']}
shared = {tp: np.nanmedian(np.stack(resid_stack[tp]), 0) for tp in ['48h','72h']}
rows = []
for tp in ['48h','72h']:
    for (t, w), res in zip(keys[tp], resid_stack[tp]):
        m = np.isfinite(res)
        r2 = res - np.nan_to_num(shared[tp])
        valid = m & np.isfinite(shared[tp])
        rows.append(dict(timepoint=tp, well=w, condition=COND[w], resid_sd_raw=np.nanstd(res[valid]),
                         resid_sd_specific=np.nanstd(r2[valid]),
                         acid_patch_specific=(ndi.gaussian_filter(np.where(valid, r2, 0), 0) > 0.06)[valid].mean()))
spec = pd.DataFrame(rows)
print("shared-pattern share of residual variance (median):",
      round(1 - (spec.resid_sd_specific**2 / spec.resid_sd_raw**2).median(), 2))
def es_tbl(df, cols):
    out = []
    for tp, d in df.groupby('timepoint'):
        pooled = d.groupby('condition')[cols].std().pow(2).mean().pow(0.5)
        out.append(((d.groupby('condition')[cols].mean() - d[d.condition=='DMSO'][cols].mean()) / pooled).add_suffix('_'+tp))
    return pd.concat(out, axis=1)
T = es_tbl(spec, ['resid_sd_raw','resid_sd_specific','acid_patch_specific'])
print(T.loc[['Fluorobrite','1uM TMZ','Baf L','Baf M','Baf H','Rot H','3BP H','Olig M','Olig H','Ace M','5uM OGM','20uM OGM','20uM Ogerin']].round(1).to_string())
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
for ax, tp in zip(axes, ['48h','72h']):
    im = ax.imshow(shared[tp], cmap='PuOr_r', vmin=-0.06, vmax=0.06); ax.axis('off'); ax.set_title(f'{tp}: median residual across all wells')
fig.colorbar(im, ax=axes, fraction=0.03, label='488/405 residual'); fig.savefig('shared_residual_pattern.svg', dpi=120, bbox_inches='tight'); plt.close(fig)

# ---- archived cell 221
print('cell 221', flush=True)

from skimage import morphology as mo
low = feat[feat.solidity < 0.92][['timepoint','well','condition','solidity']]
print(f"{len(low)} wells with solidity<0.92:", low.groupby('condition').size().to_dict())
fixed = []
for _, r in low.iterrows():
    k = (r.timepoint, r.well); m = masks[k]
    best = None
    for rad in (15, 25, 35):
        op = mo.binary_opening(m, mo.disk(rad)); L = measure.label(op)
        if L.max() == 0: continue
        reg = max(measure.regionprops(L), key=lambda x: x.area)
        cand = ndi.binary_dilation(L == reg.label, mo.disk(rad)) & m
        if cand.sum() > 0.5 * m.sum() * 0 + 1250:
            best = cand
            if measure.regionprops(cand.astype(np.uint8))[0].solidity > 0.92: break
    if best is not None and best.sum() < 0.97 * m.sum():
        masks[k] = best; fixed.append((k, m.sum()*4, best.sum()*4))
print("re-masked:", [(k[0], k[1], f"{a/1e3:.0f}k->{b/1e3:.0f}k") for k, a, b in fixed])
ziC = {tp: zproj_index(PLATES[tp]) for tp in PLATES}; riC = {}
for (tp, w), _, _ in fixed:
    zi = ziC[tp]
    if not all(c in zi.get(w, {}) for c in 'grb'):
        riC.setdefault(tp, raw_index(PLATES[tp]))
    g, r, b = load_well(tp, w, zi, riC.get(tp))
    f, ap, npf, R = well_features(g, r, masks[(tp, w)], fields[tp])
    i = feat.index[(feat.timepoint==tp) & (feat.well==w)][0]
    for kk, vv in f.items(): feat.at[i, kk] = vv
    absP[(tp, w)], normP[(tp, w)] = ap, npf
    if (tp, w) in maps48 or (tp, w) == ('48h','H6'): maps48[(tp, w)] = np.where(masks[(tp, w)], R, np.nan).astype(np.float32)
feat['mask_split'] = [ (t, w) in {k for k, _, _ in fixed} for t, w in zip(feat.timepoint, feat.well)]
print(feat[feat.condition=='Ace M'][['timepoint','well','ratio_median','resid_sd','acidic_patch_frac','solidity','mask_split']].round(3).to_string())

# ---- archived cell 225
print('cell 225', flush=True)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, w in zip(axes, ['H4','H5','H6']):
    g, r, b = load_well('48h', w, ziC['48h'], None)
    ax.imshow(b, cmap='gray', vmin=np.percentile(b, 1), vmax=np.percentile(b, 99.5))
    for c in measure.find_contours(masks[('48h', w)].astype(float), 0.5): ax.plot(c[:,1], c[:,0], 'r-', lw=0.8)
    ax.set_title(f'48h {w} (Ace M) bright field + mask'); ax.axis('off')
fig.savefig('aceM_bf_check.svg', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---- archived cell 227
print('cell 227', flush=True)

feat['qc_debris'] = (feat.condition=='Ace M') & feat.well.isin(['H5','H6'])
featQ = feat[~feat.qc_debris].copy()
featQ['family'] = featQ['condition'].map(fam)
ES = effect_sizes(featQ); ES.to_csv('spatial_effect_sizes_vs_DMSO.csv', index=False)
feat.to_csv('spatial_features_per_well.csv', index=False)

# rim/centre deltas
rows = []
for tp, d in featQ.groupby('timepoint'):
    dm = d[d.condition=='DMSO'][['rim_ratio','core_ratio']].mean()
    x = d.groupby('condition')[['rim_ratio','core_ratio']].mean() - dm; x['timepoint'] = tp; rows.append(x.reset_index())
dRC = pd.concat(rows); dRC.columns = ['condition','d_rim','d_core','timepoint']; dRC['family'] = dRC.condition.map(fam)

# overview figure on QC'd data
feat_all = feat; feat = featQ
exec(open('spatial_overview_fig.py', encoding='utf-8').read())

# maps figure: replace Ace M H6 with 20 µM Ogerin G11
g_, r_, b_ = load_well('48h', 'G11', ziC['48h'], None)
_, _, _, Rg = well_features(g_, r_, masks[('48h','G11')], fields['48h'])
maps48[('48h','G11')] = np.where(masks[('48h','G11')], Rg, np.nan)
sel = [('A1','DMSO'),('C8','Baf H'),('G8','Olig H'),('E8','3BP H'),('D8','Rot H'),('G11','20 µM Ogerin'),('D11','20 µM OGM')]
fig, axes = plt.subplots(2, len(sel), figsize=(2.1*len(sel)+1, 4.8), facecolor='black')
for j, (w, lab) in enumerate(sel):
    m = masks[('48h', w)]; R = maps48[('48h', w)]
    cy, cx = map(int, measure.regionprops(m.astype(np.uint8))[0].centroid)
    def crop(a):
        c = np.full((2*HALF, 2*HALF), np.nan); y0, y1, x0, x1 = max(0, cy-HALF), min(980, cy+HALF), max(0, cx-HALF), min(980, cx+HALF)
        c[y0-(cy-HALF):y1-(cy-HALF), x0-(cx-HALF):x1-(cx-HALF)] = a[y0:y1, x0:x1]; return c
    f_ = feat[(feat.timepoint=='48h') & (feat.well==w)].iloc[0]
    axes[0, j].imshow(crop(R), cmap=cm1, vmin=0.8, vmax=2.0); axes[0, j].set_title(lab, color='white', fontsize=8)
    axes[1, j].imshow(crop(resid_map(R, m)), cmap=cm2, vmin=-0.15, vmax=0.15)
    axes[1, j].set_title(f"acidic patches {f_.acidic_patch_frac*100:.0f}% · ξ={f_.corr_length_px:.0f}px", color='white', fontsize=6.5)
    for a in axes[:, j]: a.axis('off'); a.set_facecolor('black')
fig.subplots_adjust(left=0.01, right=0.9, wspace=0.03, hspace=0.12, top=0.86)
for row, (cm_, lo, hi, lab) in enumerate([(cm1, 0.8, 2.0, '488/405\n(higher = more acidic)'), (cm2, -0.15, 0.15, 'residual after\nradial trend')]):
    cax = fig.add_axes([0.915, 0.5 - row*0.43 + 0.03, 0.01, 0.33])
    cb = fig.colorbar(ScalarMappable(cmap=cm_, norm=Normalize(lo, hi)), cax=cax)
    cb.set_label(lab, color='white', fontsize=7); cb.ax.tick_params(colors='white', labelsize=6)
fig.suptitle('48 h: flat-field-corrected 488/405 (top) and local patchiness after removing the radial gradient (bottom; orange = more acidic than expected for that depth). Same scale in every panel.',
             color='white', fontsize=7.5)
fig.savefig('spatial_maps_48h.svg', dpi=170, facecolor='black'); plt.close(fig)

# heatmap: drop rim angular CV (optics-dominated)
show = ['ratio_median','rim_ratio','core_ratio','rim_minus_core','ratio_cv','resid_sd','acidic_patch_frac',
        'alkaline_patch_frac','corr_length_px','moran_I_12px','area_px','solidity','boundary_radius_cv']
M = np.full((len(order), 3*len(show)), np.nan)
for i, c in enumerate(order):
    for k, ft in enumerate(show):
        for t, tp in enumerate(['24h','48h','72h']):
            v = ES[(ES.condition==c) & (ES.timepoint==tp)][ft]
            if len(v): M[i, 3*k+t] = v.values[0]
fig, ax = plt.subplots(figsize=(12.5, 9))
im = ax.imshow(np.clip(M, -5, 5), cmap='RdBu_r', vmin=-5, vmax=5, aspect='auto')
for k in range(1, len(show)): ax.axvline(3*k - 0.5, color='white', lw=2)
for y in [3,6,9,12,15,18,21,24,28]: ax.axhline(y-0.5, color='white', lw=1)
ax.set_xticks([3*k+1 for k in range(len(show))]); ax.set_xticklabels([nice[s] for s in show], rotation=40, ha='right')
ax.set_yticks(range(len(order))); ax.set_yticklabels([c.replace('uM ','µM ') for c in order], fontsize=6)
ax.set_title('Spatial feature effect sizes vs same-plate DMSO (each block = 24 / 48 / 72 h; pooled-replicate SD units; red = higher). Ace M H5/H6 excluded (debris).', fontsize=7.5, loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.02); cb.set_label('effect size (SD units, clipped ±5)')
fig.savefig('spatial_feature_heatmap.svg', dpi=150, bbox_inches='tight'); plt.close(fig)
feat = feat_all
print(ES[ES.timepoint=='72h'].set_index('condition').loc[['Ace M','Ace H','Baf M','Olig M','3BP H','Rot H','20uM OGM']][['rim_minus_core','resid_sd','acidic_patch_frac','corr_length_px']].round(1).to_string())

# ---- archived cell 229
print('cell 229', flush=True)

s = open('spatial_overview_fig.py', encoding='utf-8').read().replace("('Rot H',(-30,6))", "('Rot H',(6,4))")
open('spatial_overview_fig.py', 'w', encoding='utf-8').write(s)
feat = featQ; exec(s); feat = feat_all
