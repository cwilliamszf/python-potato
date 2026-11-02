"""Run the whole pipeline in order. Usage:  python run_all.py [step ...]
e.g.  python run_all.py            (all main steps: 3 4 5 6 7 1)
      python run_all.py 1b 2       (optional QC figures and the superseded first spatial pass)
      python run_all.py 15         (acid-patch dynamics; needs step 4)
      python run_all.py 17         (bright-field texture and full-resolution surface puncta; needs step 4)
      python run_all.py 16         (primary readouts; needs steps 4 and 15, uses 17 if present)
Each step writes a log to <OUT_DIR>/logs/<step>.log and stops the run on error."""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import OUT_DIR
STEPS = {'1': 'step1_whole_spheroid.py', '1b': 'step1b_session1_qc_figures.py', '2': 'step2_legacy_spatial_v1.py',
         '3': 'step3_extract_arrays.py', '4': 'step4_spatial_features.py', '5': 'step5_statistics.py',
         '6': 'step6_spatial_figures.py', '7': 'step7_single_plane_validation.py',
         '15': 'step15_patch_dynamics.py', '16': 'step16_primary_readouts.py', '17': 'step17_texture.py'}
order = sys.argv[1:] or ['3', '4', '5', '6', '7', '1']
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
for s in order:
    log = os.path.join(OUT_DIR, 'logs', f'step{s}.log'); t0 = time.time()
    print(f'step {s}: {STEPS[s]} ...', flush=True)
    with open(log, 'w', encoding='utf-8') as fh:
        rc = subprocess.call([sys.executable, '-u', os.path.join(HERE, STEPS[s])], stdout=fh, stderr=subprocess.STDOUT)
    print(f'   exit {rc}, {time.time()-t0:.0f}s  (log: {log})', flush=True)
    if rc != 0:
        sys.exit(f'step {s} failed - see {log}')
