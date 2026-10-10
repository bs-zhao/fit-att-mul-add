import os
import pickle
import random
import sys

import numpy as np
from tqdm import tqdm
from data2param_flow import load_instance

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from c_bifood import sim_trial_aDDM, sim_trial_aRACE
from tools.funcs import load_food_data
from tools.model_info import model_infos

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

dt = 0.001
max_rt = 28.287
feature = 'trial'
sx = '1'
for_test = 0
n_round = 500
names = [f'gen{i}' for i in range(1, 21)]

# All artifacts produced by compare_regen live under food_equal/compare_regen.
# The internal layout mirrors adm-sbi/G19_v3/nn_rv1/compare_regen/save/.
output_root = '../../../../outputs/krajbich2010/compare_regen'

for m in model_names:
    base = 'mr' if for_test else 'real_data'
    dst = f'{output_root}/{base}/{feature}/s1_gen{sx}/{m}/'
    os.makedirs(dst, exist_ok=True)

    df = load_food_data('../../../../data/krajbich2010/trial_eye.csv')
    ids = list(df.subj.unique())
    info = model_infos[m]
    pnames = info['free_pnames_ml']
    ranges = info['free_pranges_ml']

    # Inputs from the main food_equal fitting pipeline remain under
    # outputs/krajbich2010; only outputs created by compare_regen are redirected.
    dec = load_instance(
        f'../../../../outputs/krajbich2010/dpsRH{sx}_dp0.15/{feature}/{m}/'
        'ParameterDecoder/parameter_decoder.pkl'
    )
    dec.batch_size = 64
    path_data = f'../../../../outputs/krajbich2010/s3_fe{sx}_maxT/real_data/fe.pkl'

    for G in names:
        gen = []
        for rd in tqdm(range(n_round), desc=f'{m} {G}'):
            i_subj = random.randint(0, len(ids) - 1)
            subj = ids[i_subj]
            ds = df[df.subj == subj]

            out = dec.predict(
                [path_data], key_param=None, post_dropout=False
            )[0].cpu().numpy().squeeze()[i_subj]
            out = np.atleast_1d(out).astype(float)

            for i in range(len(pnames)):
                out[i] = out[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]

            params = {n: float(out[i]) for i, n in enumerate(pnames)}
            for p, v in zip(info['fixed_pnames'], info['fixed_pvalues']):
                params[p] = v

            nums = np.array([ds.v0.values, ds.v1.values])
            ps, dus, cs, rts, vss, nfs = [], [], [], [], [], []

            for i, (_, r) in enumerate(ds.iterrows()):
                v = np.ascontiguousarray(nums[:, i], dtype=np.float64)
                p = np.ascontiguousarray(np.asarray(r.arr_pos, dtype=np.int64))
                du = np.ascontiguousarray(np.asarray(r.arr_du, dtype=np.int64))

                if 'aDDM' in m:
                    (c, rt, hit), sim, *_ = sim_trial_aDDM(
                        params, v, p, du, extend_last=500000, repeat=0,
                        dt=dt, rcd=1, max_rt=max_rt
                    )
                else:
                    (c, rt, hit), sim, *_ = sim_trial_aRACE(
                        params, v, p, du, extend_last=500000, repeat=0,
                        dt=dt, rcd=1, max_rt=max_rt
                    )

                if hit == 0:
                    continue

                e, a, sp, sd = sim
                ps.append(np.asarray(sp))
                dus.append(np.asarray(sd))
                cs.append(int(c))
                rts.append(float(rt))
                vss.append(v)
                nfs.append(len(sp))

            if len(cs) > 5:
                gen.append({
                    'subj': subj,
                    'arr_pos': np.array(ps, dtype=object),
                    'arr_du': np.array(dus, dtype=object),
                    'choice': np.array(cs),
                    'rt': np.array(rts),
                    'vs': np.array(vss),
                    'nf': np.array(nfs),
                    'params': params,
                })

        with open(f'{dst}/{G}.pkl', 'wb') as f:
            pickle.dump(gen, f, protocol=4)
