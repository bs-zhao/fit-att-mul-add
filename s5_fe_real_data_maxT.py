import os
import pickle
import numpy as np

from tools.funcs import load_food_data, arr2tp, get_summary

sx = '1'
dt = 0.001
eye2 = np.eye(2)

file_save = f"outputs/s3_fe{sx}_maxT/real_data/fe.pkl"
df_save = f"outputs/s3_fe{sx}_maxT/real_data/df.csv"
os.makedirs(os.path.dirname(file_save), exist_ok=True)

df = load_food_data('data/trial_eye.csv')
all_subj = list(df['subj'].unique())
df.to_csv(df_save, index=False)

fe_data = []
for subj in all_subj:
    df_subj = df[df['subj'] == subj].copy()
    real_choice = np.asarray(df_subj['choice'].values, dtype=int)
    real_rt = np.asarray(df_subj['rt'].values, dtype=float)
    all_numbers = np.array([df_subj.v0.values, df_subj.v1.values])

    dic_sub = {'subj': subj, 'choice': eye2[real_choice], 'rt': real_rt}
    lst_trialinfo, lst_fsmr = [], []
    for j in range(len(df_subj)):
        arr_pos = np.asarray(df_subj.iloc[j]['arr_pos'], dtype=np.int64)
        arr_du = np.asarray(df_subj.iloc[j]['arr_du'], dtype=np.int64)
        f_pos = arr2tp(arr_pos, arr_du)
        lst_trialinfo.append(all_numbers[:, j])
        lst_fsmr.append(np.asarray(get_summary(f_pos, dt), dtype=float))

    dic_sub['trialinfo'] = np.asarray(lst_trialinfo)
    dic_sub['fsmr'] = np.asarray(lst_fsmr)
    dic_sub['params'] = None
    fe_data.append(dic_sub)

with open(file_save, 'wb') as f:
    pickle.dump(fe_data, f, protocol=4)
print(f"saved {file_save}")
