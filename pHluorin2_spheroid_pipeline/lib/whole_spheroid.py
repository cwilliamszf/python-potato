# Step 1 per-well work: full-resolution bright-field segmentation and whole-spheroid 488/405 measurement.
# Moved out of step1_whole_spheroid.py so that worker processes can run it (lib/parallel.py). The algorithm is
# unchanged; the large-footprint morphology is computed with the exact cropped equivalents in lib/fastops.py
# (binary_dilation with disk(30) took ~7 s per well at 1960 x 1960 px).
import numpy as np, tifffile, scipy.ndimage as ndi
from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
import fastops as fo


def segment_spheroid_robust(bf_img, area_min=5_000, area_max=2_000_000):
    """Segment spheroid; fall back to no-border-clear if Otsu yields nothing."""
    inv = (bf_img.max() - bf_img.astype(np.float32))
    blurred = filters.gaussian(inv, sigma=5)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    binary = morphology.remove_small_objects(binary, min_size=1000)

    def _pick_largest(b):
        labeled = measure.label(b)
        if labeled.max() == 0:
            return None
        regions = measure.regionprops(labeled)
        largest = max(regions, key=lambda r: r.area)
        mask = fo.fill_holes(labeled == largest.label)            # == ndi.binary_fill_holes
        mask = fo.close_disk(mask, 10)                             # == morphology.binary_closing(mask, disk(10))
        return mask

    # Try with border clearing first
    mask = _pick_largest(clear_border(binary, buffer_size=5))
    # Fall back: no border clear
    if mask is None:
        mask = _pick_largest(binary)

    if mask is None:
        return None, {'qc_fail': 'no_objects', 'area_px': 0, 'equiv_diam_px': 0, 'circularity': 0}

    area = int(mask.sum())
    if area < area_min:
        return None, {'qc_fail': f'too_small', 'area_px': area, 'equiv_diam_px': 0, 'circularity': 0}
    if area > area_max:
        return None, {'qc_fail': f'too_large', 'area_px': area, 'equiv_diam_px': 0, 'circularity': 0}

    props = measure.regionprops(mask.astype(np.uint8))[0]
    equiv_diam  = props.equivalent_diameter_area
    perimeter   = props.perimeter
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    return mask, {'qc_fail': None, 'area_px': area,
                  'equiv_diam_px': float(equiv_diam), 'circularity': float(circularity)}


def measure_well(gfp, rogfp, mask, annulus_inner=10, annulus_outer=30):
    """
    Returns dict of fluorescence metrics given images and spheroid mask.
    Background estimated from an annular ring around the mask.
    """
    # Build annulus for background
    dilated_outer = fo.dilate_disk(mask, annulus_outer)           # == morphology.binary_dilation(mask, disk(30))
    dilated_inner = fo.dilate_disk(mask, annulus_inner)
    annulus = dilated_outer & ~dilated_inner

    bg_gfp   = float(np.median(gfp[annulus]))   if annulus.any() else 0.0
    bg_rogfp = float(np.median(rogfp[annulus])) if annulus.any() else 0.0

    gfp_mean   = float(gfp[mask].mean())   - bg_gfp
    rogfp_mean = float(rogfp[mask].mean()) - bg_rogfp
    gfp_total   = float(gfp[mask].sum())
    rogfp_total = float(rogfp[mask].sum())

    ratio = gfp_mean / rogfp_mean if rogfp_mean > 0 else np.nan

    return {
        'gfp_mean_raw':   float(gfp[mask].mean()),
        'rogfp_mean_raw': float(rogfp[mask].mean()),
        'bg_gfp':         bg_gfp,
        'bg_rogfp':       bg_rogfp,
        'gfp_mean_corr':  gfp_mean,
        'rogfp_mean_corr':rogfp_mean,
        'gfp_total':      gfp_total,
        'rogfp_total':    rogfp_total,
        'ratio_gfp_rogfp':ratio,
    }


FL_KEYS = ['gfp_mean_raw', 'rogfp_mean_raw', 'bg_gfp', 'bg_rogfp', 'gfp_mean_corr', 'rogfp_mean_corr',
           'gfp_total', 'rogfp_total', 'ratio_gfp_rogfp']


def _max_projection(slices):
    """Max-intensity projection of the raw z-slices [(z, path), ...] sorted by z (same values as np.stack().max(0))."""
    st = None
    for _, fp in slices:
        im = tifffile.imread(fp)
        st = im if st is None else np.maximum(st, im, out=st)
    return st


def measure_one(tp, well, condition, zfiles, raw):
    """One well. zfiles = (gfp, rogfp, bf) ZProj paths, or None -> max projection from raw = {channel: [(z, path)]}.
    Returns ('ok', record) or ('load_error', (tp, well, message)), as in the original loop."""
    try:
        if zfiles is None:
            if raw is None: raise KeyError(well)
            gfp_i, rogfp_i, bf_i = (_max_projection(raw[ch]) for ch in ['GFP', 'Oxidized roGFP2', 'Bright Field'])
        else:
            gfp_i, rogfp_i, bf_i = (tifffile.imread(f) for f in zfiles)
    except Exception as e:
        return 'load_error', (tp, well, str(e))
    mask_i, seg_props = segment_spheroid_robust(bf_i)
    rec = {'timepoint': tp, 'well': well, 'condition': condition, 'row': well[0], 'col': int(well[1:]), **seg_props}
    if mask_i is not None:
        rec.update(measure_well(gfp_i, rogfp_i, mask_i))
    else:
        rec.update({k: np.nan for k in FL_KEYS})
    return 'ok', rec
