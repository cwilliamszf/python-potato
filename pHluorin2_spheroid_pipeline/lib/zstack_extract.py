# Per-plane extraction from raw Cytation z-stacks (28 slices, 10.1 um step, z0 = autofocus plane, stack goes UP).
# Stores 4x4-binned (490x490; 1 px = 2.79 um) 488, 405 and BF stacks as float16 for a list of wells.
import numpy as np, tifffile
ZSTEP_UM = 10.1; PX_UM_FULL = 1367 / 1960; PX_UM = PX_UM_FULL * 4

def bin4(a):
    a = a.astype(np.float32); return a.reshape(490, 4, 490, 4).mean((1, 3))

def load_stack(ri, w, keys='grb'):
    """keys: which channels to read (step 7 reads only 'gr': the bright-field stack is not used there)."""
    out = {}
    for ch, key in [('GFP', 'g'), ('Oxidized roGFP2', 'r'), ('Bright Field', 'b')]:
        if key not in keys: continue
        sl = sorted(ri[w][ch])
        out[key] = np.stack([bin4(tifffile.imread(fp)) for _, fp in sl]).astype(np.float16)
    return out

def extract_wells(plate_tp, wells, ri=None, verbose=True):
    if ri is None: ri = raw_index(PLATES[plate_tp])
    S = {}
    for i, w in enumerate(wells):
        try:
            S[(plate_tp, w)] = load_stack(ri, w)
        except (PermissionError, KeyError) as e:
            print('skip', plate_tp, w, type(e).__name__)
        if verbose and (i + 1) % 6 == 0: print(plate_tp, i + 1, '/', len(wells), flush=True)
    return S
