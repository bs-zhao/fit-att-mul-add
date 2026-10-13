import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.funcs import arr2tp, get_summary
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
][:]

dt = 0.001
batch_size = 500
sx = '1'
gens = [f"gen{i}" for i in range(1, 51)]
eye2 = np.eye(2)
for_test = 1

for model_name in model_names:
    dir_gen = f"../../../outputs/krajbich2010/s1_gen{sx}/{model_name}/"
    dir_save = f"../../../outputs/krajbich2010/s3_fe{sx}/{model_name}/"
    model_gens = gens
    if for_test == 1:
        dir_gen = f"../../../outputs/krajbich2010/s1_gen{sx}_test/{model_name}/"
        dir_save = f"../../../outputs/krajbich2010/s3_fe{sx}_test/{model_name}/"
        dir_save_subjs = f"{dir_save}/subjs/"
        model_gens = ["gen1"]
        os.makedirs(dir_save_subjs, exist_ok=True)
    os.makedirs(dir_save, exist_ok=True)
    param_names = model_infos[model_name]['free_pnames_ml']

    if for_test == 1:
        count_subj = 0
        test_rows = []

    for gen in model_gens:
        with open(f"{dir_gen}/{gen}.pkl", 'rb') as f:
            raw_data = pickle.load(f)
        print(f"{model_name} {gen}: n fake subj={len(raw_data)}")

        for i_batch in range(0, len(raw_data), batch_size):
            fe_data = []
            for subj in tqdm(raw_data[i_batch:i_batch + batch_size], leave=False):
                if for_test == 1:
                    count_subj += 1
                    subj_rows = []
                fe_subj = {}
                lst_trialinfo, lst_choice, lst_rt, lst_fsmr = [], [], [], []
                for j in range(len(subj['arr_pos'])):
                    arr_pos = np.asarray(subj['arr_pos'][j], dtype=np.int64)
                    arr_du = np.asarray(subj['arr_du'][j], dtype=np.int64)
                    if len(arr_pos) == 0:
                        continue
                    f_pos = arr2tp(arr_pos, arr_du)
                    lst_trialinfo.append(np.asarray(subj['vs'][j], dtype=float))
                    lst_choice.append(eye2[int(subj['choice'][j])])
                    lst_rt.append(float(subj['rt'][j]))
                    lst_fsmr.append(np.asarray(get_summary(f_pos, dt), dtype=float))
                    if for_test == 1:
                        row = {
                            'Subject': count_subj,
                            'subj': subj['subj'],
                            'v0': float(subj['vs'][j][0]),  # right option
                            'v1': float(subj['vs'][j][1]),  # left option
                            'rt': float(subj['rt'][j]),
                            'choice': int(subj['choice'][j]),
                            'arr_pos': arr_pos.tolist(),
                            'arr_du': arr_du.tolist(),
                            'T0_ms': int(arr_du[arr_pos == 0].sum()),
                            'T1_ms': int(arr_du[arr_pos == 1].sum()),
                            'nf': int(len(arr_pos)),
                        }
                        for p in param_names:
                            row[p] = float(subj['params'][p])
                        subj_rows.append(row)
                fe_subj['trialinfo'] = np.asarray(lst_trialinfo)
                fe_subj['choice'] = np.asarray(lst_choice)
                fe_subj['rt'] = np.asarray(lst_rt)
                fe_subj['fsmr'] = np.asarray(lst_fsmr)
                fe_subj['params'] = subj['params']
                if fe_subj['trialinfo'].shape[0] > 5:
                    fe_data.append(fe_subj)
                    if for_test == 1:
                        test_rows.extend(subj_rows)
                        with open(f"{dir_save_subjs}/subj_{count_subj}.pkl", 'wb') as f:
                            pickle.dump([fe_subj], f, protocol=4)
            with open(f"{dir_save}/{gen}_batch{i_batch}.pkl", 'wb') as f:
                pickle.dump(fe_data, f, protocol=4)
            print(f"saved {dir_save}/{gen}_batch{i_batch}.pkl")

    if for_test == 1:
        columns = ['Subject', 'subj', 'v0', 'v1', 'rt', 'choice', 'arr_pos', 'arr_du', 'T0_ms', 'T1_ms', 'nf'] + list(param_names)
        df_test = pd.DataFrame(test_rows, columns=columns)
        df_test.to_csv(f"{dir_save}/fe.csv", index=False)
        df_test.drop_duplicates(subset=['Subject'], keep='first').to_csv(
            f"{dir_save}/param.csv", index=False
        )
        print(f"saved {dir_save}/fe.csv and {dir_save}/param.csv")
