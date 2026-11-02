"""Exact, faster equivalents of slow array operations used by the pipeline.

Each function returns the same array, bit for bit, as the call it replaces (tests: tools/test_fastops.py).

Cropped morphology: a binary dilation/closing/hole-filling of a mask can only change pixels within the
footprint's reach of the mask, so it is computed on the mask's bounding box plus a margin larger than that reach
and pasted back. The margin keeps a band of background pixels around the cropped result, so the crop edge cannot
influence it; where the box touches the real image edge, the crop edge is the image edge, with the same border
rule as the full-size call. Disk dilation/erosion use the exact Euclidean distance transform
(a pixel is within the disk(r) footprint of a mask pixel iff its squared distance is <= r^2; sqrt of an integer is
correctly rounded, so the comparison with r is exact). Border rules follow scikit-image 0.26 (binary_dilation:
outside = False; binary_erosion: outside = True). This also removes the dependency on the skimage binary_* wrappers,
which are deprecated in 0.26 and removed in 0.28.

nanmedian_small: numpy's own algorithm for np.nanmedian along a short axis (a masked sort, then the mean of the
two middle values), vectorised without masked-array overhead.
"""
import numpy as np, scipy.ndimage as ndi

def _box(mask, margin):
    ys = np.flatnonzero(mask.any(1)); xs = np.flatnonzero(mask.any(0))
    if len(ys) == 0: return None
    return (slice(max(ys[0] - margin, 0), min(ys[-1] + margin + 1, mask.shape[0])),
            slice(max(xs[0] - margin, 0), min(xs[-1] + margin + 1, mask.shape[1])))


def _paste(shape, sl, sub):
    out = np.zeros(shape, bool); out[sl] = sub; return out


def _edt_dilate(sub, r):
    # pixel within Euclidean distance r of a True pixel  <=>  inside the disk(r) dilation (x^2 + y^2 <= r^2)
    return ndi.distance_transform_edt(~sub) <= r


def _edt_erode(sub, r):
    # border_value=True erosion: kept unless a False pixel of the array lies within distance r
    if sub.all(): return np.ones(sub.shape, bool)
    return ndi.distance_transform_edt(sub) > r


def dilate_disk(mask, r):
    """== skimage.morphology.binary_dilation(mask, disk(r))"""
    mask = np.asarray(mask, bool); sl = _box(mask, r + 1)
    if sl is None: return np.zeros(mask.shape, bool)
    return _paste(mask.shape, sl, _edt_dilate(mask[sl], r))


def close_disk(mask, r):
    """== skimage.morphology.binary_closing(mask, disk(r))"""
    mask = np.asarray(mask, bool); sl = _box(mask, 2 * r + 2)
    if sl is None:   # dilation of nothing is nothing; the erosion then keeps nothing
        return np.zeros(mask.shape, bool)
    return _paste(mask.shape, sl, _edt_erode(_edt_dilate(mask[sl], r), r))


def dilate_iter(mask, iterations):
    """== scipy.ndimage.binary_dilation(mask, iterations=iterations)   (default 4-connected cross)"""
    mask = np.asarray(mask, bool); sl = _box(mask, iterations + 1)
    if sl is None: return np.zeros(mask.shape, bool)
    return _paste(mask.shape, sl, ndi.binary_dilation(mask[sl], iterations=iterations))


def fill_holes(mask):
    """== scipy.ndimage.binary_fill_holes(mask)"""
    mask = np.asarray(mask, bool); sl = _box(mask, 1)
    if sl is None: return np.zeros(mask.shape, bool)
    return _paste(mask.shape, sl, ndi.binary_fill_holes(mask[sl]))


def nanmedian_small(a, axis):
    """== np.nanmedian(a, axis) for a floating array (numpy uses this algorithm when a.shape[axis] < 600)."""
    a = np.asarray(a)
    s = np.sort(a, axis=axis)                                   # NaNs sort to the end
    n = np.count_nonzero(~np.isnan(a), axis=axis, keepdims=True)
    nn = np.maximum(n, 1)
    lo = np.take_along_axis(s, (nn - 1) // 2, axis=axis)       # the two middle values of the valid ones
    hi = np.take_along_axis(s, nn // 2, axis=axis)
    med = (lo + hi) / s.dtype.type(2)
    med = np.where(n > 0, med, s.dtype.type(np.nan))
    return np.squeeze(med, axis=axis)
