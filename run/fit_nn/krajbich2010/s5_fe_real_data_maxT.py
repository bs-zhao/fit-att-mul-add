from pathlib import Path as _Path
import os as _os
_os.chdir(_Path(__file__).resolve().parent)
import os
import pickle

import numpy as np

from tools.funcs import arr2tp, get_summary, load_food_data
from tools.maxT import MAX_LEN_NF, MAX_LEN_RT

sx = '1'
dt = 0.001
ds = 3
max_len_nf = MAX_LEN_NF
max_len_rt = MAX_LEN_RT

print('fixed max_len_nf:', max_len_nf)
print('fixed max_len_rt:', max_len_rt)
print('time-sequence length after downsampling:', len(np.arange(max_len_rt)[::ds]))

eye2 = np.eye(2)

file_save = f"../../../outputs/krajbich2010/s3_fe{sx}_maxT/real_data/fe.pkl"
df_save = f"../../../outputs/krajbich2010/s3_fe{sx}_maxT/real_data/df.csv"
os.makedirs(os.path.dirname(file_save), exist_ok=True)

df = load_food_data('../../../outputs/krajbich2010/s0_prepare_data/trial_eye.csv')
all_subj = list(df['subj'].unique())
df.to_csv(df_save, index=False)

fe_data = []
n_skipped = 0
skipped_trials = []

for subj in all_subj:
    df_subj = df[df['subj'] == subj].copy()

    fe_subj = {'subj': subj}
    lst_trialinfo, lst_choice, lst_rt, lst_fsmr = [], [], [], []

    for row_idx, row in df_subj.iterrows():
        arr_pos = np.asarray(row['arr_pos'], dtype=np.int64)
        arr_du = np.asarray(row['arr_du'], dtype=np.int64)
        f_pos = arr2tp(arr_pos, arr_du)

        exceed_nf = len(arr_pos) >= max_len_nf
        exceed_rt = np.ceil(len(f_pos) / ds) >= np.floor(max_len_rt / ds)
        if exceed_nf or exceed_rt:
            n_skipped += 1
            skipped_trials.append({
                'subj': subj,
                'row': int(row_idx),
                'rt': float(row['rt']),
                'rt_ms': int(len(f_pos)),
                'nf': int(len(arr_pos)),
                'reason': '+'.join([
                    x for x, hit in [('nf', exceed_nf), ('rt', exceed_rt)] if hit
                ]),
            })
            continue

        choice = int(row['choice'])
        values = np.asarray([row['v0'], row['v1']], dtype=float)
        fsmr = np.asarray(get_summary(f_pos, dt), dtype=float)

        lst_trialinfo.append(values)
        lst_choice.append(eye2[choice])
        lst_rt.append(float(row['rt']))
        lst_fsmr.append(fsmr)

    fe_subj['trialinfo'] = np.asarray(lst_trialinfo, dtype=float)
    fe_subj['choice'] = np.asarray(lst_choice, dtype=float)
    fe_subj['rt'] = np.asarray(lst_rt, dtype=float)
    fe_subj['fsmr'] = np.asarray(lst_fsmr, dtype=float)
    fe_subj['params'] = None

    if fe_subj['trialinfo'].shape[0] > 5:
        fe_data.append(fe_subj)

with open(file_save, 'wb') as f:
    pickle.dump(fe_data, f, protocol=4)

print(f'saved {file_save}')
print(f'skipped trials exceeding fixed maxT: {n_skipped}')
if skipped_trials:
    print('skipped trial details:')
    for x in skipped_trials:
        print(
            f"  subj={x['subj']} row={x['row']} "
            f"rt={x['rt']:.3f}s rt_ms={x['rt_ms']} "
            f"nf={x['nf']} reason={x['reason']}"
        )
