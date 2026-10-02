# Build fixed-format features from recovery-test simulations created with
# s1_gen1w.py after setting for_test=1.
import os
import pickle
import numpy as np
from tools.funcs import arr2tp, get_summary
from tools.model_info import model_names

dt = 0.001
eye2 = np.eye(2)

for model_name in model_names:
    src = f"outputs/s1_gen1_test/{model_name}/gen1.pkl"
    if not os.path.exists(src):
        continue
    with open(src, 'rb') as f:
        raw_data = pickle.load(f)
    fe_data = []
    for subj in raw_data:
        d = {'subj': subj['subj'], 'choice': eye2[np.asarray(subj['choice'], dtype=int)],
             'rt': np.asarray(subj['rt'], dtype=float), 'params': subj['params']}
        d['trialinfo'] = np.asarray(subj['vs'], dtype=float)
        d['fsmr'] = np.asarray([get_summary(arr2tp(p, du), dt) for p, du in zip(subj['arr_pos'], subj['arr_du'])])
        fe_data.append(d)
    out = f"outputs/s3_fe1_maxT/test/{model_name}/fe.pkl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(fe_data, f, protocol=4)
    print(f"saved {out}")
