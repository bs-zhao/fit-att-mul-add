from pathlib import Path as _Path
import os as _os
_SCRIPT_DIR = _Path(__file__).resolve().parent
_os.chdir(_SCRIPT_DIR)
import os
import pickle
import sys

import numpy as np

sys.path.append(str(_SCRIPT_DIR.parent))
from tools.funcs import arr2tp, get_summary
from tools.maxT import MAX_LEN_NF, MAX_LEN_RT

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]


dt = 0.001
ds = 3
feature = 'trial'
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--for-test', action='store_true')
ap.add_argument('--models', nargs='+', choices=model_names, default=model_names)
ap.add_argument('--n-gen', type=int, default=20)
args = ap.parse_args()
for_test = int(args.for_test)
model_names = args.models
sx = '1'
max_len_nf = MAX_LEN_NF
max_len_rt = MAX_LEN_RT
eye2 = np.eye(2)
output_root = '../../../../outputs/krajbich2010/compare_regen'

print('fixed max_len_nf:', max_len_nf)
print('fixed max_len_rt:', max_len_rt)
print('time-sequence length after downsampling:', len(np.arange(max_len_rt)[::ds]))

for m in model_names:
    base = 'mr' if for_test else 'real_data'
    src = f'{output_root}/{base}/{feature}/s1_gen{sx}/{m}/'
    dst = f'{output_root}/{base}/{feature}/s3_fe{sx}_maxT/{m}/'
    os.makedirs(dst, exist_ok=True)
    n_skipped = 0

    for g in [f'gen{i}' for i in range(1, args.n_gen + 1)]:
        file_in = f'{src}/{g}.pkl'
        if not os.path.exists(file_in):
            print(f'skip missing {file_in}')
            continue

        with open(file_in, 'rb') as f:
            raw = pickle.load(f)

        fe = []
        for s in raw:
            fe_subj = {'subj': s.get('subj')}
            lst_trialinfo, lst_choice, lst_rt, lst_fsmr = [], [], [], []

            for j in range(len(s['arr_pos'])):
                arr_pos = np.asarray(s['arr_pos'][j], dtype=np.int64)
                arr_du = np.asarray(s['arr_du'][j], dtype=np.int64)
                f_pos = arr2tp(arr_pos, arr_du)

                if (
                    len(arr_pos) >= max_len_nf
                    or np.ceil(len(f_pos) / ds) >= np.floor(max_len_rt / ds)
                ):
                    n_skipped += 1
                    continue

                choice = int(s['choice'][j])
                values = np.asarray(s['vs'][j], dtype=float)
                fsmr = np.asarray(get_summary(f_pos, dt), dtype=float)

                lst_trialinfo.append(values)
                lst_choice.append(eye2[choice])
                lst_rt.append(float(s['rt'][j]))
                lst_fsmr.append(fsmr)

            fe_subj['trialinfo'] = np.asarray(lst_trialinfo, dtype=float)
            fe_subj['choice'] = np.asarray(lst_choice, dtype=float)
            fe_subj['rt'] = np.asarray(lst_rt, dtype=float)
            fe_subj['fsmr'] = np.asarray(lst_fsmr, dtype=float)
            fe_subj['params'] = s['params']

            if fe_subj['trialinfo'].shape[0] > 5:
                fe.append(fe_subj)

        with open(f'{dst}/{g}_batch0.pkl', 'wb') as f:
            pickle.dump(fe, f, protocol=4)
        print(f'saved {dst}/{g}_batch0.pkl')

    print(f'{m}: skipped trials exceeding fixed maxT: {n_skipped}')
