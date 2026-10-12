# Binary food-choice version of adm-sbi/G19_v3/nn_rv1/s1_gen1w.py

import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from c_bifood import sim_trial_aDDM, sim_trial_aRACE
from tools.model_info import model_infos
from tools.funcs import load_food_data, make_params

dt = 0.001
max_rt = 14.5

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

for_test = 0
n_round = 500
names = [f"gen{i}" for i in range(1, 51)]

for model_name in model_names:
    N_GEN = len(names)
    dir_save = f"../../../outputs/food_equal/s1_gen1/{model_name}/"
    if for_test == 1:
        names = ["gen1"]
        N_GEN = 1
        dir_save = f"../../../outputs/food_equal/s1_gen1_test/{model_name}/"
    os.makedirs(dir_save, exist_ok=True)

    df = load_food_data('../../../data/food_equal/trial_eye.csv')
    all_subj = list(df['subj'].unique())
    model_info = model_infos[model_name]

    rt_all = []
    for GEN in names:
        print(f"{model_name}: {GEN}/{N_GEN}")
        gen_rlt = []
        test_rows = [] if for_test == 1 else None

        for rd in tqdm(range(n_round)):
            subj = random.choice(all_subj)
            df_subj = df[df['subj'] == subj]
            arr_pos = df_subj['arr_pos'].values
            arr_du = df_subj['arr_du'].values
            all_numbers = np.array([df_subj.v0.values, df_subj.v1.values])
            n_trials = len(df_subj)
            params = make_params(model_info)

            dic_sub = {"subj": subj, "arr_pos": [], "arr_du": [],
                       "choice": [], "rt": [], "vs": []}
            arr_pos_list, arr_du_list, nf_list = [], [], []
            choice_list, rt_list, vs_list = [], [], []

            for i in range(n_trials):
                numbers = np.ascontiguousarray(all_numbers[:, i], dtype=np.float64)
                arr_pos_trial = np.ascontiguousarray(np.asarray(arr_pos[i], dtype=np.int64))
                arr_du_trial = np.ascontiguousarray(np.asarray(arr_du[i], dtype=np.int64))

                if 'aDDM' in model_name:
                    (choice, rt, hit), info, *_ = sim_trial_aDDM(
                        params, numbers, arr_pos_trial, arr_du_trial,
                        extend_last=500000, repeat=0, dt=dt, rcd=1, max_rt=max_rt)
                    e, a, sim_arr_pos, sim_arr_du = info
                else:
                    (choice, rt, hit), info, *_ = sim_trial_aRACE(
                        params, numbers, arr_pos_trial, arr_du_trial,
                        extend_last=500000, repeat=0, dt=dt, rcd=1, max_rt=max_rt)
                    e, a, sim_arr_pos, sim_arr_du = info

                if hit == 0:
                    continue
                arr_pos_list.append(np.asarray(sim_arr_pos))
                arr_du_list.append(np.asarray(sim_arr_du))
                nf_list.append(len(sim_arr_pos))
                choice_list.append(int(choice))
                rt_list.append(float(rt))
                vs_list.append(numbers)

            if len(choice_list) <= 5:
                continue
            dic_sub['arr_pos'] = np.array(arr_pos_list, dtype=object)
            dic_sub['arr_du'] = np.array(arr_du_list, dtype=object)
            dic_sub['nf'] = np.array(nf_list)
            dic_sub['choice'] = np.array(choice_list)
            dic_sub['rt'] = np.array(rt_list)
            dic_sub['vs'] = np.array(vs_list)
            dic_sub['params'] = params
            gen_rlt.append(dic_sub)
            rt_all += rt_list

            if for_test == 1:
                # Each accepted synthetic subject has one parameter draw.
                synthetic_id = len(gen_rlt)
                for j in range(len(choice_list)):
                    pos = np.asarray(arr_pos_list[j], dtype=np.int64)
                    du = np.asarray(arr_du_list[j], dtype=np.int64)
                    row = {
                        'Subject': synthetic_id,
                        'subj': subj,
                        'v0': float(vs_list[j][0]),  # right option
                        'v1': float(vs_list[j][1]),  # left option
                        'rt': float(rt_list[j]),
                        'choice': int(choice_list[j]),
                        'arr_pos': pos.tolist(),
                        'arr_du': du.tolist(),
                        'T0_ms': int(du[pos == 0].sum()),
                        'T1_ms': int(du[pos == 1].sum()),
                        'nf': int(nf_list[j]),
                    }
                    for p in model_info['free_pnames']:
                        row[p] = float(params[p])
                    test_rows.append(row)

        save_path = f"{dir_save}/{GEN}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(gen_rlt, f, protocol=4)
        print(f"Successfully saved {save_path}")

        if for_test == 1:
            columns = ['Subject', 'subj', 'v0', 'v1', 'rt', 'choice', 'arr_pos', 'arr_du', 'T0_ms', 'T1_ms', 'nf'] + list(model_info['free_pnames'])
            df_test = pd.DataFrame(test_rows, columns=columns)
            df_test.to_csv(f"{dir_save}/{GEN}.csv", index=False)
            df_test.drop_duplicates(subset=['Subject'], keep='first').to_csv(
                f"{dir_save}/param.csv", index=False
            )
            print(f"saved {dir_save}/{GEN}.csv and {dir_save}/param.csv")
