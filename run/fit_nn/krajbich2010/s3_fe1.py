from pathlib import Path as _Path
import os as _os
_os.chdir(_Path(__file__).resolve().parent)
import os
import pickle
import numpy as np
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
]

dt = 0.001
batch_size = 500
sx = '1'
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', choices=model_names, default=model_names)
ap.add_argument('--n-gen', type=int, default=50)
args = ap.parse_args()
model_names = args.models
gens = [f"gen{i}" for i in range(1, args.n_gen + 1)]
eye2 = np.eye(2)

for model_name in model_names:
    dir_gen = f"../../../outputs/krajbich2010/s1_gen{sx}/{model_name}/"
    dir_save = f"../../../outputs/krajbich2010/s3_fe{sx}/{model_name}/"
    os.makedirs(dir_save, exist_ok=True)
    param_names = model_infos[model_name]['free_pnames_ml']

    for gen in gens:
        with open(f"{dir_gen}/{gen}.pkl", 'rb') as f:
            raw_data = pickle.load(f)
        print(f"{model_name} {gen}: n fake subj={len(raw_data)}")

        for i_batch in range(0, len(raw_data), batch_size):
            fe_data = []
            for subj in tqdm(raw_data[i_batch:i_batch + batch_size], leave=False):
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
                fe_subj['trialinfo'] = np.asarray(lst_trialinfo)
                fe_subj['choice'] = np.asarray(lst_choice)
                fe_subj['rt'] = np.asarray(lst_rt)
                fe_subj['fsmr'] = np.asarray(lst_fsmr)
                fe_subj['params'] = subj['params']
                if fe_subj['trialinfo'].shape[0] > 5:
                    fe_data.append(fe_subj)
            with open(f"{dir_save}/{gen}_batch{i_batch}.pkl", 'wb') as f:
                pickle.dump(fe_data, f, protocol=4)
            print(f"saved {dir_save}/{gen}_batch{i_batch}.pkl")
