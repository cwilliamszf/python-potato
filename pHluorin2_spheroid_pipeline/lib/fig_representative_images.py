import fastops as fo   # exact cropped equivalents of the slow full-resolution morphology
import matplotlib; matplotlib.rcParams['svg.fonttype'] = 'none'
import tifffile, scipy.ndimage as ndi
from skimage import filters, morphology, measure
from skimage.segmentation import clear_border
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
g0, r0_, b0 = ld('A1')
lims = {k: np.percentile(a, [1, 99]) for k, a in zip('grb', (g0, r0_, b0))}
nrm = lambda a, k: np.clip((a.astype(float)-lims[k][0])/(lims[k][1]-lims[k][0]+1e-9), 0, 1)
RLO, RHI = 1.0, 1.9
fig, axes = plt.subplots(2, 4, figsize=(15, 8)); axes = axes.ravel()
for ax, (w, lab) in zip(axes, rep):
    try:
        g, r, b = ld(w)
    except PermissionError:
        ax.text(0.5, 0.5, f'{lab}\nfile locked', ha='center', va='center', transform=ax.transAxes); ax.axis('off'); continue
    m = seg(b)
    ratio = np.where(r > 0, g.astype(float)/r.astype(float), np.nan)   # 488/405
    p = measure.regionprops(m.astype(np.uint8))[0]; cy, cx = map(int, p.centroid)
    pad = max(int(p.equivalent_diameter_area*0.375)+80, 200)
    y0, y1, x0, x1 = max(0,cy-pad), min(1960,cy+pad), max(0,cx-pad), min(1960,cx+pad)
    bn, gn, rn = nrm(b,'b'), nrm(g,'g'), nrm(r,'r')
    rgb = np.stack([np.clip(bn*.5+rn*.7,0,1), np.clip(bn*.5+gn*.7,0,1), bn*.5], -1)[y0:y1, x0:x1]
    mc = m[y0:y1, x0:x1]
    ax.imshow(rgb); ax.imshow(np.where(mc, ratio[y0:y1, x0:x1], np.nan), cmap='RdBu_r', vmin=RLO, vmax=RHI, alpha=0.5)
    for c in measure.find_contours(mc.astype(float), 0.5): ax.plot(c[:,1], c[:,0], 'w-', lw=1.2)
    ax.text(0.02, 0.98, lab, transform=ax.transAxes, fontsize=8, color='white', fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55))
    ax.text(0.02, 0.04, f'488/405 = {np.nanmean(ratio[m]):.2f}   area = {m.sum()/1000:.0f}k px²', transform=ax.transAxes,
            fontsize=6, color='white', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.45))
    ax.axis('off')
cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
cb = fig.colorbar(ScalarMappable(cmap='RdBu_r', norm=Normalize(RLO, RHI)), cax=cax)
cb.set_label('488/405 ratio\n(higher = more acidic)', fontsize=7)
fig.suptitle('Representative spheroids, 48 h — RGB: bright field (grey), 488-ex (green), 405-ex (magenta); overlay: 488/405 within spheroid mask', fontsize=8, y=1.0)
plt.tight_layout(rect=[0, 0, 0.91, 1]); fig.savefig('representative_images.svg', dpi=130, bbox_inches='tight'); plt.close(fig)
