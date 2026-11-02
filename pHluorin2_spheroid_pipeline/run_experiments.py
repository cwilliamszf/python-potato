"""Run the follow-up experiments (experiments/*.json): one extraction pass (step 3) that stores both the max projection
and the equatorial single plane, then steps 4-6 on each.  Usage: python run_experiments.py [experiment ...]
Outputs: <SPHEROID_RESULTS_ROOT>/<experiment>_max and <experiment>_equatorial (default root: results_followup)."""
import os, sys, subprocess, time, glob
HERE = os.path.dirname(os.path.abspath(__file__))
exps = sys.argv[1:] or [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(HERE, 'experiments', '*.json')))]
ROOT = os.environ.get('SPHEROID_RESULTS_ROOT', os.path.join(HERE, 'results_followup'))
STEP = {'3': 'step3_extract_arrays.py', '4': 'step4_spatial_features.py', '5': 'step5_statistics.py', '6': 'step6_spatial_figures.py'}

def run(nm, proj, s, extra=None):
    out = os.path.join(ROOT, f'{nm}_{proj}'); os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    env = dict(os.environ, SPHEROID_EXPERIMENT=os.path.join(HERE, 'experiments', nm + '.json'), SPHEROID_OUT_DIR=out,
               SPHEROID_PROJECTION=proj, PYTHONIOENCODING='utf-8', **(extra or {}))
    t0 = time.time(); log = os.path.join(out, 'logs', f'step{s}.log')
    with open(log, 'w', encoding='utf-8') as fh:
        rc = subprocess.call([sys.executable, '-u', os.path.join(HERE, STEP[s])], stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(nm, proj, 'step', s, 'exit', rc, f'{time.time()-t0:.0f}s', flush=True)
    if rc != 0:
        print('   see', log, flush=True)
    return rc

for nm in exps:
    if not os.path.exists(os.path.join(ROOT, f'{nm}_max', 'stage1_arrays.npz')) or os.environ.get('RECOMPUTE'):
        if run(nm, 'max', '3', {'SPHEROID_ALT_OUT': os.path.join(ROOT, f'{nm}_equatorial')}): continue
    for proj in ['max', 'equatorial']:
        for s in ['4', '5', '6']:
            if run(nm, proj, s): break
