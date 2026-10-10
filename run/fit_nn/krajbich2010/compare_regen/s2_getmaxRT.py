import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

model_names = [
    "aRACE_t",
    "aRACE_g",
]

feature = 'trial'
sx = '1'
for_test = 0
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
