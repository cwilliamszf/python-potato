# Full-plate single-plane validation. For every spheroid:
#  - chromatic shift (405 slice best matched to each 488 slice), equatorial plane (sharpest 488 outline)
#  - depth series of chromatic-MATCHED plane ratio (488 slice z, 405 slice z+dz) at rim / core / whole
#  - max-projection source planes per channel
#  - full spatial feature set on (a) max projection, (b) matched equatorial plane, (c) UNmatched equatorial plane
import numpy as np, pandas as pd, scipy.ndimage as ndi, json, os, time
import fastops as fo
DZ_PLATE = 2          # plate-level shift used for features (pilot: 2-3 slices); per-well estimate also recorded

def robust_het(res):
    res = res.astype(np.float32); V = np.isfinite(res)
    if V.sum() < 50: return np.nan, np.nan
    v = res[V]; mad = 1.4826 * np.median(np.abs(v - np.median(v)))
    return float(mad), corr_length(np.where(V, np.clip(res, -5 * mad, 5 * mad), 0), V)

def feats2d(g, r, m, F, lg, lr, prefix):
    o, mp = well_features(g, r, m, F, (lg, lr))
    keep = {}
    for base in ['ratio_median', 'rim_ratio', 'core_ratio', 'rim_minus_core', 'acid_patch_frac', 'patch_boundary_enrich']:
        keep[f'{prefix}{base}'] = o.get(base + '_cal', np.nan)
    sd, cl = robust_het(mp['_cal']['res'])
    keep[f'{prefix}resid_sd_rob'] = sd; keep[f'{prefix}corr_len_rob'] = cl
    return keep, mp['_cal']

def zwell(tp, w, stk, m, F):
    nz = stk['g'].shape[0]
    d = ndi.distance_transform_edt(m); nd = d / d.max()
    far = ~fo.dilate_iter(m, 20)                      # == ~ndi.binary_dilation(m, iterations=20)
    rim = m & (nd <= 0.2) & (d > 3); core = nd > 0.6; whole = m & (d > 3)
    din = d; dout = ndi.distance_transform_edt(~m); ring = ((din > 0) & (din <= 4)) | ((dout > 0) & (dout <= 4))
    gz = stk['g'].astype(np.float32); rz = stk['r'].astype(np.float32)
    lg = np.array([np.median(gz[z][far]) for z in range(nz)]); lr = np.array([np.median(rz[z][far]) for z in range(nz)])
    edge = []
    for z in range(nz):
        gy, gx = np.gradient(ndi.gaussian_filter(gz[z], 1.0)); edge.append(np.mean(np.hypot(gy, gx)[ring]))
    edge = ndi.uniform_filter1d(np.array(edge), 3)
    dzw, _ = chroma_shift(stk, m)
    z_eq = int(np.argmax(edge[:nz - DZ_PLATE]))
    rec = dict(timepoint=tp, well=w, condition=COND[w], dz_well=dzw, z_eq=z_eq, R_um=float(np.sqrt(m.sum() / np.pi) * PX_UM))
    # depth series, matched and unmatched
    prof = {}
    for tag, dz in [('m', DZ_PLATE), ('u', 0)]:
        for reg_n, reg in [('rim', rim), ('core', core), ('whole', whole)]:
            prof[f'{tag}_{reg_n}'] = np.full(nz, np.nan)
        for z in range(nz - DZ_PLATE):
            R, _, _ = cal_ratio(gz[z], rz[z + dz], F, lg[z], lr[z + dz])
            for reg_n, reg in [('rim', rim), ('core', core), ('whole', whole)]:
                prof[f'{tag}_{reg_n}'][z] = np.median(R[reg])
    # max-projection source planes
    ag = np.argmax(gz, 0); ar = np.argmax(rz, 0)
    for reg_n, reg in [('rim', rim), ('core', core)]:
        rec[f'mp_src488_{reg_n}'] = float(np.median(ag[reg])); rec[f'mp_src405_{reg_n}'] = float(np.median(ar[reg]))
    # features: max projection (from same stack, binned), matched equatorial, unmatched equatorial
    gmp, rmp = gz.max(0), rz.max(0)
    f1, _ = feats2d(gmp, rmp, m, F, np.median(gmp[far]), np.median(rmp[far]), 'mp_')
    f2, mp_eq = feats2d(gz[z_eq], rz[z_eq + DZ_PLATE], m, F, lg[z_eq], lr[z_eq + DZ_PLATE], 'eq_')
    f3, _ = feats2d(gz[z_eq], rz[z_eq], m, F, lg[z_eq], lr[z_eq], 'equ_')
    rec.update(f1); rec.update(f2); rec.update(f3)
    rec['eq_sbr488'] = float(np.median(gz[z_eq][m]) / lg[z_eq]); rec['eq_sbr405'] = float(np.median(rz[z_eq + DZ_PLATE][m]) / lr[z_eq + DZ_PLATE])
    maps = dict(R=mp_eq['R'], res=mp_eq['res'], centroid=mp_eq['centroid'], eq_r=mp_eq['eq_r'])
    return rec, prof, maps

def zwell_job(tp, w, ri_w, m):
    """Worker entry point (lib/parallel.py): read one spheroid's 488/405 stacks and run zwell.
    Plate fields come from PAR_FIELDS (sent once per worker)."""
    try:
        # the original read the bright-field stack too (unused here) and skipped the well if that failed;
        # keep the skip rule without reading the pixels: channel present and every slice file readable
        bf = ({w: ri_w} if ri_w is not None else {})[w]['Bright Field']
        for _, fp in bf:
            with open(fp, 'rb'): pass
        stk = load_stack({w: ri_w}, w, keys='gr')
    except Exception as e:
        return 'skip', type(e).__name__
    return 'ok', zwell(tp, w, stk, m, PAR_FIELDS[tp])

ZWELL_LIBS = ['common.py', 'stage2_features.py', 'zstack_extract.py', 'zstack_analysis.py', 'zfull.py']

def run_all(ri_by_tp, out_csv='zplane_features.csv'):
    """Every spheroid, in parallel worker processes (SPHEROID_WORKERS); rows, profiles and maps are collected in
    plate/well order, so the outputs are identical to a serial run. Resumes from out_csv as before."""
    from parallel import WellPool
    rows, PROFS, MAPS_EQ = [], {}, {}
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv); rows = prev.to_dict('records'); done = set(zip(prev.timepoint, prev.well))
    t0 = time.time(); n = 0
    tasks = [(tp, w, ri_by_tp[tp].get(w), M[(tp, w)]) for tp in TPS for w in WELLS
             if (tp, w) in M and (tp, w) not in EXCL and (tp, w) not in done]
    with WellPool(HERE, ZWELL_LIBS, shared={'PAR_FIELDS': fields}, fallback_ns=globals()) as pool:
        for (tp, w, _, _), (status, val) in zip(tasks, pool.imap('zwell_job', tasks)):
            k = (tp, w)
            if status == 'skip':
                print('skip', k, val, flush=True); continue
            rec, prof, mp = val
            rows.append(rec); PROFS[k] = prof; MAPS_EQ[k] = mp; n += 1
            if n % 24 == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f'{n} wells, {time.time()-t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows), PROFS, MAPS_EQ
