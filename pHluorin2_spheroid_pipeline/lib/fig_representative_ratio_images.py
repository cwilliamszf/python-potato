import fastops as fo   # exact cropped equivalents of the slow full-resolution morphology
import matplotlib; matplotlib.rcParams['svg.fonttype'] = 'none'

import numpy as np, tifffile, scipy.ndimage as ndi, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')

Z = zproj_dir('48h')
def ld(w):
    return [tifffile.imread(os.path.join(Z, f"{w}_-2_{i}_1_ZProj[{n}]_001.tif")) for i, n in
            [(1,'GFP 469,525'),(2,'Oxidized roGFP2 400,525'),(3,'Bright Field')]]
def seg(bf):
    b = filters.gaussian(bf.max()-bf.astype(np.float32), sigma=5); b = b > filters.threshold_otsu(b)
    b = morphology.remove_small_objects(b, min_size=1000)
    for cand in (clear_border(b, buffer_size=5), b):
        L = measure.label(cand)
        if L.max():
            r = max(measure.regionprops(L), key=lambda r: r.area)
            return fo.close_disk(fo.fill_holes(L==r.label), 10)   # == binary_closing(binary_fill_holes(.), disk(10))
    return None

rep = [('A1','DMSO'),('A8','1 µM TMZ'),('C8','Baf H'),('D8','Rot H'),
       ('E8','3BP H'),('G8','Olig H'),('D11','20 µM OGM'),('H11','50 µM Ogerin')]
HALF = 420   # fixed 840x840 px crop for every panel -> same spatial scale
panels = []
for w, lab in rep:
    try:
        g, r, b = ld(w)
    except PermissionError:
        panels.append((lab, None, np.nan)); continue
    m = seg(b)
    gs, rs = ndi.gaussian_filter(g.astype(float), 1.5), ndi.gaussian_filter(r.astype(float), 1.5)
    ratio = np.where(m & (rs > 0), gs/np.maximum(rs, 1), np.nan)          # 488/405
    cy, cx = map(int, measure.regionprops(m.astype(np.uint8))[0].centroid)
    canvas = np.full((2*HALF, 2*HALF), np.nan)
    y0, y1, x0, x1 = max(0, cy-HALF), min(1960, cy+HALF), max(0, cx-HALF), min(1960, cx+HALF)
    canvas[y0-(cy-HALF):y1-(cy-HALF), x0-(cx-HALF):x1-(cx-HALF)] = ratio[y0:y1, x0:x1]
    panels.append((lab, canvas, float(np.nanmean(g[m]/1.0)/np.nanmean(r[m]))))

pool = np.concatenate([p[1][np.isfinite(p[1])] for p in panels if p[1] is not None])
lo, hi = np.percentile(pool, [1, 99]); print(f"display range 488/405: {lo:.2f}-{hi:.2f}")
lo, hi = np.floor(lo*10)/10, np.ceil(hi*10)/10

cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('black')
fig, axes = plt.subplots(2, 4, figsize=(14, 7.4), facecolor='black')
for ax, (lab, img, mr) in zip(axes.ravel(), panels):
    ax.set_facecolor('black'); ax.axis('off')
    if img is None:
        ax.text(0.5, 0.5, f'{lab}\n(file locked)', color='white', ha='center', transform=ax.transAxes); continue
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(f'{lab}   (mean 488/405 = {mr:.2f})', color='white', fontsize=9, pad=3)
cax = fig.add_axes([0.93, 0.2, 0.012, 0.6])
cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(lo, hi)), cax=cax)
cb.set_label('488/405  (higher = more acidic)', color='white', fontsize=9)
cb.ax.tick_params(colors='white', labelsize=8); cb.outline.set_edgecolor('white')
fig.suptitle('48 h — 488/405 ratio images, masked to spheroid (all panels same scale and LUT range)',
             color='white', fontsize=10, y=0.99)
fig.subplots_adjust(left=0.01, right=0.91, top=0.92, bottom=0.02, wspace=0.03, hspace=0.12)
fig.savefig('representative_ratio_images.svg', dpi=200, facecolor='black')
plt.close(fig)
