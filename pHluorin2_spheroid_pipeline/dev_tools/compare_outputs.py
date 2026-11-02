"""Compare every output file of two pipeline result folders.  usage: compare_outputs.py <dirA> <dirB>
CSV: exact (NaN == NaN); otherwise the max absolute numeric difference is reported. NPZ/PKL/JSON: deep equality.
SVG: text after removing matplotlib's date stamp and its random element ids. Logs are skipped."""
import os, re, sys, json, pickle, numpy as np, pandas as pd

A, B = sys.argv[1], sys.argv[2]
SKIP_DIRS = {'logs'}

def files(root):
    out = set()
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if x not in SKIP_DIRS]
        for f in fs: out.add(os.path.relpath(os.path.join(d, f), root))
    return out

def deq(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(deq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(deq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and \
            (np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc') if a.dtype.kind != 'O' else all(deq(x, y) for x, y in zip(a.ravel(), b.ravel())))
    if isinstance(a, pd.DataFrame):
        return a.equals(b)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    try:
        return bool(a == b)
    except Exception:
        return False

def csv_cmp(pa, pb):
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    if a.equals(b): return 'identical'
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f'DIFFERENT shape/columns {a.shape} vs {b.shape}'
    worst, where = 0.0, None
    for c in a.columns:
        x, y = a[c], b[c]
        if x.equals(y): continue
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            d = (x - y).abs(); both_nan = x.isna() & y.isna(); one_nan = x.isna() ^ y.isna()
            if one_nan.any(): return f'DIFFERENT NaN pattern in {c}'
            m = float(d[~both_nan].max() / max(1e-300, float(x.abs().max())))
            if m > worst: worst, where = m, c
        else:
            return f'DIFFERENT non-numeric column {c}'
    return f'numeric diff, max relative {worst:.2e} in {where}'

SVG_RE = [re.compile(r'<dc:date>.*?</dc:date>', re.S), re.compile(r'(id|href|clip-path|xlink:href)="[^"]*"'),
          re.compile(r'url\(#[^)]*\)')]

def svg_norm(p):
    s = open(p, encoding='utf-8').read()
    for r in SVG_RE: s = r.sub('', s)
    return s

fa, fb = files(A), files(B)
bad = 0
for f in sorted(fa | fb):
    if f not in fa or f not in fb:
        print(f'{"MISSING in " + ("A" if f not in fa else "B"):20s} {f}'); bad += 1; continue
    pa, pb = os.path.join(A, f), os.path.join(B, f); ext = os.path.splitext(f)[1].lower()
    try:
        if ext == '.csv': r = csv_cmp(pa, pb)
        elif ext == '.npz':
            za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
            r = 'identical' if za.files == zb.files and all(deq(za[k], zb[k]) for k in za.files) else 'DIFFERENT'
        elif ext == '.pkl':
            r = 'identical' if deq(pickle.load(open(pa, 'rb')), pickle.load(open(pb, 'rb'))) else 'DIFFERENT'
        elif ext == '.json':
            r = 'identical' if deq(json.load(open(pa)), json.load(open(pb))) else 'DIFFERENT'
        elif ext == '.svg':
            r = 'identical' if svg_norm(pa) == svg_norm(pb) else 'DIFFERENT (svg text)'
        else:
            r = 'identical' if open(pa, 'rb').read() == open(pb, 'rb').read() else 'DIFFERENT (bytes)'
    except Exception as e:
        r = f'ERROR {e!r}'
    if r != 'identical': bad += 1
    print(f'{r:40s} {f}')
print(f'\n{len(fa | fb)} files, {bad} not identical')
