# Stage 1: one I/O pass. For every well x timepoint: segment spheroid on 2x2-binned bright field,
# then store 4x4-binned (490x490; 1 px = 4 full-res px) 488 and 405 intensity + mask.
# CONVENTION downstream: ratio = 488/405, HIGHER = MORE ACIDIC.
exec(open(LIB('common.py'), encoding='utf-8').read())   # PLATES, COND, WELLS, zproj_index, raw_index, load_well, seg
from skimage import morphology as mo

def bin2f(a):
    a = a.astype(np.float32); return a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).mean((1, 3))

def trim_debris(m490):
    """Remove attached debris from low-solidity masks by opening; keep largest blob."""
    p = measure.regionprops(m490.astype(np.uint8))[0]
    if p.solidity >= 0.92: return m490, False
    best = None
    for rad in (4, 6, 9):
        op = mo.binary_opening(m490, mo.disk(rad)); L = measure.label(op)
        if L.max() == 0: continue
        reg = max(measure.regionprops(L), key=lambda x: x.area)
        cand = ndi.binary_dilation(L == reg.label, mo.disk(rad)) & m490
        if cand.sum() >= 80:
            best = cand
            if measure.regionprops(cand.astype(np.uint8))[0].solidity > 0.92: break
    if best is not None and best.sum() < 0.97 * m490.sum(): return best, True
    return m490, False

GP, RP = {}, {}   # equatorial-plane arrays (v2 loader only)

def stage1_well(tp, w, zi_w, ri_w):
    """One well: load, segment, bin. zi_w = {'g'|'r'|'b': ZProj path}, ri_w = {channel: [(z, path)]} or None.
    Returns (status record, 488, 405, mask or None, equatorial 488, equatorial 405 or None)."""
    rec = dict(timepoint=tp, well=w, condition=COND[w], status='ok', mask_split=False)
    zi = {w: zi_w}; ri = {w: ri_w} if ri_w is not None else {}
    gp16 = rp16 = m4 = None
    if SEGMENTATION == 'v2':
        g, r, b, zsel, gp, rp = load_well_v2(w, ri); rec['z_plane'] = zsel
        gp16 = bin2f(gp).astype(np.float16); rp16 = bin2f(rp).astype(np.float16)
    elif PROJECTION == 'max':
        g, r, b = load_well(tp, w, zi, ri)          # 980x980 binned max projection
    else:
        g, r, b, zsel = load_well_plane(w, ri); rec['z_plane'] = zsel
    g16 = bin2f(g).astype(np.float16); r16 = bin2f(r).astype(np.float16)
    m = seg_v2(b) if SEGMENTATION == 'v2' else seg(b)
    if m is None:
        rec['status'] = 'seg_fail'
    else:
        m4 = bin2f(m) > 0.5
        m4 = ndi.binary_fill_holes(m4)
        m4, split = trim_debris(m4); rec['mask_split'] = split
    return rec, g16, r16, m4, gp16, rp16


def run_stage1():
    """All wells of all plates, in parallel worker processes (lib/parallel.py); same order and values as serial."""
    from parallel import WellPool
    G, Rr, M, rows = {}, {}, {}, []
    tasks = []
    for tp, plate in PLATES.items():
        zi = zproj_index(plate); ri = None
        for w in WELLS:
            if (PROJECTION != 'max' or SEGMENTATION == 'v2' or not all(k in zi.get(w, {}) for k in 'grb')) and ri is None:
                ri = raw_index(plate)
            tasks.append((tp, w, zi.get(w, {}), ri.get(w) if ri is not None else None))
    with WellPool(HERE, ['stage1_extract.py'], fallback_ns=globals()) as pool:
        for (tp, w, _, _), (rec, g16, r16, m4, gp16, rp16) in zip(tasks, pool.imap('stage1_well', tasks)):
            if gp16 is not None:
                GP[(tp, w)] = gp16; RP[(tp, w)] = rp16
            G[(tp, w)] = g16; Rr[(tp, w)] = r16
            if m4 is not None:
                M[(tp, w)] = m4
            rows.append(rec)
            if w == WELLS[-1]: print(tp, 'done', flush=True)
    return G, Rr, M, pd.DataFrame(rows)


# ---- single-focal-plane alternative to the max projection (SPHEROID_PROJECTION=equatorial)
PROJECTION = os.environ.get('SPHEROID_PROJECTION', 'max')

def _edge_ring(m, band=8):
    d_in = ndi.distance_transform_edt(m); d_out = ndi.distance_transform_edt(~m)
    return ((d_in > 0) & (d_in <= band)) | ((d_out > 0) & (d_out <= band))

def _edge_sharp(img, m, band=8, ring=None):
    if ring is None: ring = _edge_ring(m, band)      # pass ring when scoring many slices against the same mask
    gy, gx = np.gradient(ndi.gaussian_filter(img, 2.0))
    return float(np.mean(np.hypot(gy, gx)[ring]))

def load_well_plane(w, ri):
    """Equatorial plane = the z-slice with the sharpest 488 outline around the bright-field (max-projection) mask.
    488 and 405 are taken from the same slice (the ~20 um 405 chromatic focal shift is below the z-step of the
    coarse stacks; in the fine-z pilot matched and unmatched planes gave equivalent features). Bright field = max
    projection (used only for segmentation). Returns 980x980 2x2-binned g, r, b and the chosen slice index."""
    st = {}
    for ch, k in CHANNELS:
        st[k] = [bin2(tifffile.imread(fp)) for _, fp in sorted(ri[w][ch])]
    b = np.max(np.stack(st['b']), 0)
    m = seg(b)
    if m is None:
        zsel = len(st['g']) // 2
    else:
        ring = _edge_ring(m); zsel = int(np.argmax([_edge_sharp(gz, m, ring=ring) for gz in st['g']]))
    return st['g'][zsel], st['r'][zsel], b, zsel


# ---- follow-up experiments (coarse 7-slice stacks, smaller spheroids, strong phase halos in the bright-field max
# projection): segmentation on the bright-field MIN projection (config SEGMENTATION = 'v2')
SEGMENTATION = os.environ.get('SPHEROID_SEGMENTATION', globals().get('SEGMENTATION_CFG', 'v1'))

def seg_v2(b):
    """b = 980x980 bright-field min projection. Flatten by a large Gaussian, threshold dark objects (Otsu, capped at
    0.92 x local background), open/close/fill, keep the dark object closest to the image centre (spheroids sit at the
    bottom of the round well), weighted towards compact objects. Returns a bool mask or None."""
    flat = b / ndi.gaussian_filter(b, 60)
    sm = ndi.gaussian_filter(flat, 2)
    core = sm[(sm > np.percentile(sm, 0.5)) & (sm < np.percentile(sm, 99.5))]
    m = sm < min(filters.threshold_otsu(core), 0.92)
    m = mo.binary_opening(m, mo.disk(2)); m = ndi.binary_fill_holes(mo.binary_closing(m, mo.disk(4)))
    L = measure.label(m)
    regs = [x for x in measure.regionprops(L) if 600 <= x.area <= 400_000]
    if not regs: return None
    cy, cx = np.array(m.shape) / 2
    reg = min(regs, key=lambda x: np.hypot(x.centroid[0] - cy, x.centroid[1] - cx) / (np.sqrt(x.area) + 1) - 0.5 * x.solidity)
    return fo.fill_holes(fo.close_disk(L == reg.label, 6))   # == binary_fill_holes(binary_closing(., disk(6)))

def load_well_v2(w, ri):
    """Read every slice once. 488/405: max projection (PROJECTION='max') or the equatorial slice (sharpest 488
    outline). Bright field: min projection (for seg_v2). Returns 980x980 g, r, b and the equatorial slice index."""
    st = {}
    for ch, k in CHANNELS:
        st[k] = np.stack([bin2(tifffile.imread(fp)) for _, fp in sorted(ri[w][ch])])
    b = st['b'].min(0)
    m = seg_v2(b)
    if m is not None:
        ring = _edge_ring(m); zsel = int(np.argmax([_edge_sharp(gz, m, ring=ring) for gz in st['g']]))
    else:
        zsel = st['g'].shape[0] // 2
    gp, rp = st['g'][zsel], st['r'][zsel]
    if PROJECTION == 'max':
        return st['g'].max(0), st['r'].max(0), b, zsel, gp, rp
    return gp, rp, b, zsel, gp, rp
