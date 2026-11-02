"""Exactness tests for lib/fastops.py against the scikit-image / scipy / numpy calls they replace.
usage: python dev_tools/test_fastops.py      (expects 0 mismatches)"""
import os, sys, time, warnings, numpy as np, scipy.ndimage as ndi
warnings.filterwarnings('ignore')
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import fastops as fo
from skimage import morphology as mo
rng = np.random.default_rng(0)
def rand_mask(n, kind):
    yy, xx = np.mgrid[:n, :n]
    if kind == 'empty': return np.zeros((n, n), bool)
    if kind == 'full': return np.ones((n, n), bool)
    cy, cx = rng.uniform(-0.1, 1.1, 2) * n; R = rng.uniform(2, n / 2.5)
    m = (yy - cy) ** 2 + (xx - cx) ** 2 < R ** 2
    m ^= rng.random((n, n)) < 0.01                             # speckle + holes
    if kind == 'holes': m &= ~(((yy - cy) ** 2 + (xx - cx) ** 2) < (R / 3) ** 2)
    if kind == 'edge': m[:, :3] = True
    return m
bad = 0; N = 0
for n in (61, 240, 490):
    for kind in ['blob', 'holes', 'edge', 'empty', 'full'] * 6:
        m = rand_mask(n, kind)
        for r in (1, 3, 5, 10):
            N += 1
            bad += not np.array_equal(fo.dilate_disk(m, r), mo.binary_dilation(m, mo.disk(r)))
            bad += not np.array_equal(fo.close_disk(m, r), mo.binary_closing(m, mo.disk(r)))
        for it in (1, 6, 20, 40):
            bad += not np.array_equal(fo.dilate_iter(m, it), ndi.binary_dilation(m, iterations=it))
        bad += not np.array_equal(fo.fill_holes(m), ndi.binary_fill_holes(m))
print('morphology cases:', N, 'mismatches:', bad)
# nanmedian
badm = 0
for dt in (np.float32, np.float64):
    for shp, ax in [((245, 245, 4), 2), ((245, 245, 16), 2), ((40, 245, 245), 0), ((3, 7), 1), ((96, 490, 490), 0)]:
        for frac in (0, 0.3, 0.8, 1.0):
            a = rng.normal(size=shp).astype(dt) * 10
            a[rng.random(shp) < frac] = np.nan
            if shp[0] > 50: a[:, :5] = np.nan                   # all-NaN slices
            x, y = np.nanmedian(a, axis=ax), fo.nanmedian_small(a, ax)
            ok = x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
            badm += not ok
print('nanmedian mismatches:', badm)
# timing at production size
m = rand_mask(1960, 'blob'); m[:] = False; yy, xx = np.mgrid[:1960, :1960]; m = (yy - 980) ** 2 + (xx - 1000) ** 2 < 380 ** 2
for name, f, g in [('dilate disk30', lambda: mo.binary_dilation(m, mo.disk(30)), lambda: fo.dilate_disk(m, 30)),
                   ('close disk10', lambda: mo.binary_closing(m, mo.disk(10)), lambda: fo.close_disk(m, 10)),
                   ('fill_holes', lambda: ndi.binary_fill_holes(m), lambda: fo.fill_holes(m))]:
    t = time.perf_counter(); A = f(); t1 = time.perf_counter(); B = g(); t2 = time.perf_counter()
    print(f'{name:14s} original {1000*(t1-t):7.1f} ms   fast {1000*(t2-t1):6.1f} ms   equal {np.array_equal(A, B)}')
a = rng.normal(size=(245, 245, 4)).astype(np.float32); a[a > 1] = np.nan
t = time.perf_counter(); np.nanmedian(a, axis=2); t1 = time.perf_counter(); fo.nanmedian_small(a, 2); t2 = time.perf_counter()
print(f'nanmedian 245x245x4 original {1000*(t1-t):.1f} ms fast {1000*(t2-t1):.1f} ms')
