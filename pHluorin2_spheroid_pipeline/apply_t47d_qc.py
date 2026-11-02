"""T47D_exp5 QC policy, applied to both projections' spatial_features_v2.csv after step 4.

Why this plate needs its own rule: T47D forms markedly less round spheroids than MCF7 - no vehicle
spheroid reaches the MCF7 roundness gate (circularity 0.734-0.789 vs the 0.80 cut; edge roughness
0.077-0.153 vs the 0.08 cut) - so the absolute gate used for the MCF7 plates would reject 263 of 288
spheroid-timepoints including every vehicle. Shape is therefore REPORTED (vehicle-referenced robust
z over circularity/solidity/edge roughness/convexity) and used as a sensitivity gate, not as the
primary exclusion. The primary exclusions are the three failure modes that make a ratio unreadable:

  low_signal    either channel below 1.15x the local medium (pipeline rule, LOW_SIGNAL_SBR)
  seg_fail      phase-contrast mask disagrees with the fluorescence reference mask (Dice < 0.50)
  clip_severe   fewer than 70% of the mask's pixels survive removal of detector-clipped bins.

REVISION (clipping): the original policy excluded a whole spheroid when >5% of its mask was within
5% of the 65504 ceiling in the 405 channel. That was measured and found to be far too strict. The
readout is a MEDIAN over tens of thousands of mask pixels, which is robust to a minority of clipped
pixels: recomputing each well's median with the clipped bins removed moves it by <=2.5% for every
bafilomycin well and <=6.2% for every well on the plate except dorsomorphin (A7, +10.1% at 72 h,
where only 48% of the mask survives). The 5% rule discarded 17 of 288 well-reads for a bias smaller
than the vehicle replicate spread, and in one case (bafilomycin 20 nM at 72 h) it removed the two
normal-brightness wells and kept a dim outlier, manufacturing a spurious 2.28x ratio from n=1.

The clipping correction is now applied at PIXEL level, not well level:
  ratio_median_cal <- ratio_median_cal * (median over mask minus clipped bins) / (median over mask)
and a well is excluded only when keep_frac < 0.70, i.e. when too little of the spheroid remains to
trust any statistic. A clipped bin is one where the 4x4-binned 488 or 405 value is >= 0.90 of the
ceiling. Note the direction of the bias: clipping the 405 denominator can only push 488/405 UP, so a
clipped well reading BELOW vehicle cannot be explained by clipping.

Adds columns: convexity, axis_ratio, shape_z_t47d, dice_v2_fl, sat405_frac, sat488_frac,
seg_fail, clip405, not_round_t47d (shape_z_t47d > 3), qc_exclude_signal_only, qc_exclude.
Usage: python apply_t47d_qc.py <qc_scan.csv> <features.csv> [<features.csv> ...]
"""
import sys, numpy as np, pandas as pd
from skimage import measure, morphology as mo

SHAPE_Z_MAX = 3.0
DICE_MIN = 0.50
KEEP_MIN = 0.70          # minimum fraction of the mask surviving clipped-bin removal
CLIP_BIN = 0.90          # a 4x4-binned pixel at >= this fraction of the ceiling counts as clipped
SAT = 65504


def shape_metrics(npz_path):
    """convexity / axis ratio per spheroid-timepoint from the stored stage-1 masks."""
    z = np.load(npz_path)
    rows = []
    for i, k in enumerate(z['keys']):
        tp, w = str(k).split('|')
        m = z['m'][i].astype(bool)
        p = measure.regionprops(m.astype(np.uint8))[0]
        ph = measure.regionprops(mo.convex_hull_image(m).astype(np.uint8))[0]
        rows.append(dict(timepoint=tp, well=w, convexity=ph.perimeter / p.perimeter,
                         axis_ratio=(p.axis_minor_length / p.axis_major_length) if p.axis_major_length else np.nan))
    return pd.DataFrame(rows)


def clip_correction(npz_path, maps_pkl):
    """Per well-read: fraction of the mask surviving clipped-bin removal, and the multiplicative
    correction to the median calibrated ratio obtained by dropping those bins."""
    import pickle
    z = np.load(npz_path)
    with open(maps_pkl, 'rb') as fh:
        mp = pickle.load(fh)
    rows = []
    for i, k in enumerate(z['keys']):
        tp, w = str(k).split('|')
        m = z['m'][i].astype(bool)
        R = np.asarray(mp[(tp, w)]['R'], dtype=np.float32)
        clip = (z['r'][i] >= CLIP_BIN * SAT) | (z['g'][i] >= CLIP_BIN * SAT)
        keep = m & ~clip & np.isfinite(R)
        med_all = np.nanmedian(np.where(m & np.isfinite(R), R, np.nan))
        med_clean = np.nanmedian(np.where(keep, R, np.nan)) if keep.sum() > 50 else np.nan
        rows.append(dict(timepoint=tp, well=w, keep_frac=keep.sum() / max(m.sum(), 1),
                         clip_factor=(med_clean / med_all) if med_all else np.nan))
    return pd.DataFrame(rows)


def apply_qc(feat_csv, qc, shp):
    F = pd.read_csv(feat_csv)
    F = F.drop(columns=[c for c in ('convexity', 'axis_ratio') if c in F], errors='ignore').merge(
        shp, on=['timepoint', 'well'], how='left')
    F = F.merge(qc[['timepoint', 'well', 'dice_v2_fl', 'sat405_frac', 'sat488_frac']], on=['timepoint', 'well'], how='left')
    F = F.drop(columns=[c for c in ('keep_frac', 'clip_factor') if c in F], errors='ignore').merge(
        clip_correction(feat_csv.replace('spatial_features_v2.csv', 'stage1_arrays.npz'),
                        feat_csv.replace('spatial_features_v2.csv', 'spatial_maps_cal.pkl')),
        on=['timepoint', 'well'], how='left')
    ref = F[F.condition == 'DMSO']
    zs = []
    for c, sign in [('circularity', -1), ('solidity', -1), ('edge_roughness', +1), ('convexity', -1)]:
        med = ref[c].median(); mad = 1.4826 * np.median(np.abs(ref[c] - med))
        zs.append(sign * (F[c] - med) / max(mad, 1e-9))
    F['shape_z_t47d'] = pd.concat(zs, axis=1).max(axis=1)
    F['not_round_t47d'] = F.shape_z_t47d > SHAPE_Z_MAX
    F['seg_fail'] = F.dice_v2_fl < DICE_MIN
    F['clip_severe'] = F.keep_frac < KEEP_MIN
    F['ratio_median_cal_uncorrected'] = F.ratio_median_cal
    F['ratio_median_cal'] = (F.ratio_median_cal * F.clip_factor).fillna(F.ratio_median_cal)
    F['qc_exclude_signal_only'] = F.low_signal
    F['qc_exclude'] = F.low_signal | F.seg_fail | F.clip_severe
    F.to_csv(feat_csv, index=False)
    print(feat_csv, '| low_signal', int(F.low_signal.sum()), 'seg_fail', int(F.seg_fail.sum()),
          'clip_severe', int(F.clip_severe.sum()), '-> qc_exclude', int(F.qc_exclude.sum()),
          '| not_round_t47d', int(F.not_round_t47d.sum()), flush=True)


if __name__ == '__main__':
    qc = pd.read_csv(sys.argv[1]).rename(columns={'r_satfrac': 'sat405_frac', 'g_satfrac': 'sat488_frac'})
    shp = shape_metrics(sys.argv[2].replace('spatial_features_v2.csv', 'stage1_arrays.npz'))
    for f in sys.argv[2:]:
        apply_qc(f, qc, shp)
