import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.funcs import arr2tp, get_summary

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]

dt = 0.001
feature = 'trial'
for_test = 0
sx = '1'
eye2 = np.eye(2)
output_root = '../../../../outputs/food_equal/compare_regen'

for m in model_names:
    base = 'mr' if for_test else 'real_data'
    src = f'{output_root}/{base}/{feature}/s1_gen{sx}/{m}/'
    dst = f'{output_root}/{base}/{feature}/s3_fe{sx}_maxT/{m}/'
    os.makedirs(dst, exist_ok=True)

    for g in [f'gen{i}' for i in range(1, 21)]:
        with open(f'{src}/{g}.pkl', 'rb') as f:
            raw = pickle.load(f)

        fe = [
            {
                'trialinfo': np.asarray(s['vs'], dtype=float),
                'choice': eye2[np.asarray(s['choice'], dtype=int)],
                'rt': np.asarray(s['rt'], dtype=float),
                'fsmr': np.asarray([
                    get_summary(arr2tp(p, du), dt)
                    for p, du in zip(s['arr_pos'], s['arr_du'])
                ], dtype=float),
                'params': s['params'],
            }
            for s in raw if len(s['rt']) > 5
        ]

        with open(f'{dst}/{g}_batch0.pkl', 'wb') as f:
            pickle.dump(fe, f, protocol=4)
