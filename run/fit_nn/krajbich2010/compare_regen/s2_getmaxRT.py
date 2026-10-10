from pathlib import Path as _Path
import os as _os
_SCRIPT_DIR = _Path(__file__).resolve().parent
_os.chdir(_SCRIPT_DIR)
import os
import pickle
import sys

import numpy as np

sys.path.append(str(_SCRIPT_DIR.parent))

model_names = [
    "aDDM_1",
    "aDDM_2",
    "aDDM_t",
    "aDDM_g",
    "aRACE_1",
    "aRACE_2",
    "aRACE_t",
    "aRACE_g",
]

feature = 'trial'
sx = '1'
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--for-test', action='store_true')
args = ap.parse_args()
for_test = int(args.for_test)
output_root = '../../../../outputs/krajbich2010/compare_regen'

for m in model_names:
    base = 'mr' if for_test else 'real_data'
    rts, nfs = [], []
    for g in [f'gen{i}' for i in range(1, 6)]:
        with open(f'{output_root}/{base}/{feature}/s1_gen{sx}/{m}/{g}.pkl', 'rb') as f:
            raw = pickle.load(f)
        for e in raw:
            rts += list(e['rt'])
            nfs += list(e['nf'])
    print(m, np.percentile(rts, 99.75), np.percentile(nfs, 99.75))
