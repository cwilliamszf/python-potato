"""Compare step-15 patch-dynamics output with the ground truth of make_synth_patches.py.
usage: python validate_patch_dynamics.py <results_dir> <truth.json>   -> patch_validation.csv (+ printed table)"""
import sys, json, numpy as np, pandas as pd
R, TRUTHF = sys.argv[1], sys.argv[2]
T = json.load(open(TRUTHF)); TR = pd.read_csv(f'{R}/patch_transitions.csv'); SP = pd.read_csv(f'{R}/patch_per_spheroid.csv')
TPS = ['24h', '48h', '72h']; cons = ['24h->48h', '48h->72h']
rows = []
for w, t in T.items():
    ph, par = t['phenotype'], t['param']
    for p in cons:
        a, b = p.split('->'); ea = t['events'].get(b, {})
        truth = dict(n1=len(t['timepoints'][a]['patches']), n2=len(t['timepoints'][b]['patches']),
                     emerged=len(ea.get('emerged', [])), vanished=len(ea.get('vanished', [])),
                     merged=len(ea.get('merged', [])), split=len(ea.get('split', [])))
        ids_a = {q['id']: q for q in t['timepoints'][a]['patches']}; ids_b = {q['id']: q for q in t['timepoints'][b]['patches']}
        both = [i for i in ids_a if i in ids_b]
        truth['disp'] = float(np.mean([np.hypot(ids_b[i]['u'] - ids_a[i]['u'], ids_b[i]['v'] - ids_a[i]['v']) for i in both])) if both else np.nan
        truth['rel_growth'] = float(np.mean([2 * np.log2(ids_b[i]['s'] / ids_a[i]['s']) for i in both])) if both else np.nan
        truth['rotation'] = t['timepoints'][b]['rotation_deg'] - t['timepoints'][a]['rotation_deg']
        for reg in ['fixed', 'rotreg']:
            m = TR[(TR.well == w) & (TR.pair == p) & (TR.register == reg)]
            if m.empty: continue
            m = m.iloc[0]
            rows.append(dict(well=w, condition=t['condition'], phenotype=ph, param=par, pair=p, register=reg,
                             **{f'true_{k}': v for k, v in truth.items()},
                             meas_n1=m.n1, meas_n2=m.n2, meas_emerged=m.n_emerged, meas_vanished=m.n_vanished,
                             meas_merged=m.n_merged, meas_split=m.n_split, meas_disp=m.disp_median,
                             meas_rel_growth=m.rel_growth_patch_mean, meas_rotation=((m.rotation_deg + 180) % 360) - 180,
                             jaccard_mask=m.jaccard_mask_dist, centroid_jaccard=m.centroid_jaccard_dist))
V = pd.DataFrame(rows); V.to_csv(f'{R}/patch_validation.csv', index=False)
f = V[(V.register == 'fixed') & (V.phenotype != 'rotate') | (V.register == 'rotreg') & (V.phenotype == 'rotate')]   # rotating spheroids: after rotation registration
g = f.groupby(['phenotype', 'param'], dropna=False)
S = pd.DataFrame({'n': g.size(),
    'patches t1 true/meas': g.true_n1.mean().round(1).astype(str) + ' / ' + g.meas_n1.mean().round(1).astype(str),
    'count exact %': (100 * g[['true_n1', 'meas_n1', 'true_n2', 'meas_n2']].apply(lambda d: ((d.true_n1 == d.meas_n1) & (d.true_n2 == d.meas_n2)).mean())).round(0),
    'disp true/meas (R)': g.true_disp.mean().round(3).astype(str) + ' / ' + g.meas_disp.mean().round(3).astype(str),
    'rel growth true/meas (log2)': g.true_rel_growth.mean().round(2).astype(str) + ' / ' + g.meas_rel_growth.mean().round(2).astype(str),
    'emerged true/meas': g.true_emerged.mean().round(2).astype(str) + ' / ' + g.meas_emerged.mean().round(2).astype(str),
    'vanished true/meas': g.true_vanished.mean().round(2).astype(str) + ' / ' + g.meas_vanished.mean().round(2).astype(str),
    'merged t/m': g.true_merged.mean().round(2).astype(str) + '/' + g.meas_merged.mean().round(2).astype(str),
    'split t/m': g.true_split.mean().round(2).astype(str) + '/' + g.meas_split.mean().round(2).astype(str),
    'mask Jaccard d': g.jaccard_mask.mean().round(2), 'centroid Jaccard d': g.centroid_jaccard.mean().round(2)})
pd.set_option('display.width', 250); print(S.to_string())
r = V[V.phenotype == 'rotate'].pivot_table(index=['param'], columns='register', values=['meas_disp', 'meas_rotation'], aggfunc='mean')
print('\nrotation nuisance (truth: patches fixed in the spheroid):'); print(r.round(3).to_string())

# ---- per patch: match each designed patch to the nearest detected patch (<= 0.15 R) and check acidity / brightness flags
import pickle, os
PTB = pd.read_csv(f'{R}/patch_table.csv'); M = pickle.load(open(f'{R}/spatial_maps_cal.pkl', 'rb'))
prow = []
for w, t in T.items():
    for tp, d in t['timepoints'].items():
        if (tp, w) not in M: continue
        mp = M[(tp, w)]; cy, cx = mp['centroid']; rr = float(mp['eq_r'])
        rot = np.deg2rad(d['rotation_deg']); c, s = np.cos(rot), np.sin(rot)
        meas = PTB[(PTB.timepoint == tp) & (PTB.well == w)]
        for p in d['patches']:
            u, v = p['u'] * c + p['v'] * s, -p['u'] * s + p['v'] * c
            un = ((d['centre'][0] + u * d['radius']) / 4 - cy) / rr; vn = ((d['centre'][1] + v * d['radius']) / 4 - cx) / rr
            dist = np.hypot(meas.u - un, meas.v - vn) if len(meas) else pd.Series(dtype=float)
            hit = meas.loc[dist.idxmin()] if len(meas) and dist.min() <= 0.15 else None
            prow.append(dict(well=w, timepoint=tp, phenotype=t['phenotype'], kind=p.get('kind', 'acid'), found=hit is not None,
                             **({k: hit[k] for k in ['ratio_median', 'log_contrast_ring', 'log_contrast_depth', 'lres_peak',
                                                     'sig488_log2_vs_depth', 'sig405_log2_vs_depth', 'signal_loss']} if hit is not None else {})))
PV = pd.DataFrame(prow); PV['signal_loss'] = PV.signal_loss.astype(float); PV.to_csv(f'{R}/patch_validation_per_patch.csv', index=False)
print('\nper designed patch (all timepoints):')
print(PV.groupby('kind').agg(n=('found', 'size'), detected=('found', 'mean'), flagged_signal_loss=('signal_loss', 'mean'),
      contrast_depth=('log_contrast_depth', 'median'), contrast_ring=('log_contrast_ring', 'median'),
      sig488=('sig488_log2_vs_depth', 'median'), sig405=('sig405_log2_vs_depth', 'median')).round(3).to_string())
fp = PTB.merge(PV[PV.found][['well', 'timepoint']].drop_duplicates(), on=['well', 'timepoint'])
print('flagged patches with no designed patch nearby:', int(PTB.signal_loss.sum() - PV.signal_loss.fillna(0).sum()))

# ---- step 17: surface puncta and bright-field necrosis; step 15 alkaline definition; step 16 growth / signal
if os.path.exists(f'{R}/puncta_table.csv'):
    PU = pd.read_csv(f'{R}/puncta_table.csv'); TXF = pd.read_csv(f'{R}/texture_features.csv')
    pr = []
    for w, t in T.items():
        for tp, d in t['timepoints'].items():
            det = PU[(PU.timepoint == tp) & (PU.well == w)]
            for q in d.get('puncta', []):
                dist = np.hypot(det.y - q['y'], det.x - q['x']) if len(det) else pd.Series(dtype=float)
                hit = det.loc[dist.idxmin()] if len(det) and dist.min() <= 3 else None
                pr.append(dict(kind_true=q['kind'], amp=q['amp'], depth_um=q['depth_px'] * 1367 / 1960, found=hit is not None,
                               kind_meas=hit['kind'] if hit is not None else None))
    PR = pd.DataFrame(pr)
    if len(PR):
        PR['amp_bin'] = pd.cut(PR.amp, [0.1, 0.25, 0.4, 0.6, 1.0])
        print('\nsurface puncta: recall by designed amplitude (fraction of extra signal at the centre)')
        print(PR.groupby(['kind_true', 'amp_bin'], observed=True).found.mean().unstack().round(2).to_string())
        f = PR[PR.found]
        print('classification of detected puncta (rows = designed kind):')
        print(pd.crosstab(f.kind_true, f.kind_meas, normalize='index').round(2).to_string())
        fp = PU[~PU.condition.isin(['5uM Ogerin', '20uM Ogerin'])]
        tot_sph = TXF[~TXF.condition.isin(['5uM Ogerin', '20uM Ogerin'])].shape[0]
        print(f'false puncta in the {tot_sph} spheroid-timepoints without designed puncta: {len(fp)} '
              f'({len(fp) / tot_sph:.2f} per spheroid; {int((fp.kind == "acidic").sum())} classed acidic)')
    nec = TXF[TXF.condition == '50uM Ogerin'].groupby('timepoint')[['bf_core_vs_rim_log2', 'dark_core_frac', 'bf_texture_core', 'bf_glcm_contrast']].median()
    nec['designed_core_darkening_log2'] = [np.log2(1 - v) for v in [0, 0.2, 0.35][:len(nec)]]   # in the designed core zone
    oth = TXF[TXF.condition != '50uM Ogerin'].groupby('timepoint')[['bf_core_vs_rim_log2', 'dark_core_frac', 'bf_texture_core', 'bf_glcm_contrast']].median()
    print('\nbright-field necrotic core (50uM Ogerin) vs all other spheroids (medians):'); print(nec.round(3).to_string()); print(oth.round(3).to_string())
if os.path.exists(f'{R}/patch_per_spheroid_alk.csv'):
    SA = pd.read_csv(f'{R}/patch_per_spheroid_alk.csv'); Fs = pd.read_csv(f'{R}/spatial_features_v2.csv')
    m = SA.merge(Fs[['timepoint', 'well', 'alk_patch_frac_cal']], on=['timepoint', 'well'])
    dlt = (m.patch_frac - m.alk_patch_frac_cal).abs()
    print(f'\nalkaline patches rebuilt in step 15 vs step 4 alk_patch_frac_cal: max |difference| {dlt.max():.4f}, '
          f'identical in {(dlt < 1e-6).mean():.0%} of {len(m)} spheroid-timepoints')
if os.path.exists(f'{R}/primary_readouts_per_spheroid.csv'):
    D = pd.read_csv(f'{R}/primary_readouts_per_spheroid.csv')
    print('\ngrowth (designed log2(1.12^2) = 0.327 per day for every spheroid) and total signal per day, by condition (median):')
    print(D.groupby('condition')[['growth_log2_per_day', 'signal_log2_per_day']].median().loc[['DMSO', 'Rot H', 'Baf L', 'Baf M', 'Baf H', '50uM Ogerin']].round(3).to_string())
