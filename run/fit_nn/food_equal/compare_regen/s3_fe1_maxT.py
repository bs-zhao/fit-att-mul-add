import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.funcs import arr2tp, build_sequence_with_strength, get_summary
from tools.maxT import MAX_LEN_NF, MAX_LEN_RT

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
]

dt = 0.001
ds = 3
feature = 'trial'
for_test = 0
sx = '1'
pad_value_nf = -1
pad_value_rt = -1
max_len_nf = MAX_LEN_NF
max_len_rt = MAX_LEN_RT
eye2 = np.eye(2)
eye6 = np.eye(6)
output_root = '../../../../outputs/food_equal/compare_regen'

print('fixed max_len_nf:', max_len_nf)
print('fixed max_len_rt:', max_len_rt)
print('time-sequence length after downsampling:', len(np.arange(max_len_rt)[::ds]))

for m in model_names:
    base = 'mr' if for_test else 'real_data'
    src = f'{output_root}/{base}/{feature}/s1_gen{sx}/{m}/'
    dst = f'{output_root}/{base}/{feature}/s3_fe{sx}_maxT/{m}/'
    os.makedirs(dst, exist_ok=True)
    n_skipped = 0

    for g in [f'gen{i}' for i in range(1, 21)]:
        file_in = f'{src}/{g}.pkl'
        if not os.path.exists(file_in):
            print(f'skip missing {file_in}')
            continue

        with open(file_in, 'rb') as f:
            raw = pickle.load(f)

        fe = []
        for s in raw:
            fe_subj = {'subj': s.get('subj')}
            lst_trialinfo, lst_choice, lst_rt = [], [], []
            lst_fsmr, lst_pad_posdu, lst_pad_posduchoice = [], [], []
            lst_pad_fpos_onehot2, lst_pad_nlp1, lst_pad_nlp2, lst_pad_nlp1v = [], [], [], []

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

                arr_pos_onehot = eye2[arr_pos]
                arr_posdu = np.concatenate(
                    [arr_pos_onehot, arr_du.reshape(-1, 1)], axis=-1
                )
                arr_posdu_pad = np.pad(
                    arr_posdu,
                    ((0, max_len_nf - len(arr_pos)), (0, 0)),
                    mode='constant',
                    constant_values=pad_value_nf,
                )

                arr_posduchoice = eye6[arr_pos]
                arr_posduchoice[:, 2] = arr_du
                arr_posduchoice[:, -1] = values[arr_pos]
                add_row = np.zeros(6)
                add_row[3 + choice] = 1
                arr_posduchoice = np.concatenate(
                    [arr_posduchoice, add_row.reshape(1, -1)], axis=0
                )
                arr_posduchoice_pad = np.pad(
                    arr_posduchoice,
                    ((0, max_len_nf - len(arr_pos)), (0, 0)),
                    mode='constant',
                    constant_values=pad_value_nf,
                )

                f_pos_pad = np.pad(
                    f_pos,
                    (0, max_len_rt - len(f_pos)),
                    constant_values=pad_value_rt,
                )[::ds]
                len_valid = int(np.sum(f_pos_pad != pad_value_rt))

                nlp2_pad = build_sequence_with_strength(
                    f_pos_pad, choice, values, broadcast_choice=True
                )
                nlp1_pad = nlp2_pad.copy()
                nlp1_pad[:len_valid, -2:] = 0

                nlp1v_pad = nlp1_pad.copy()
                add_col = np.sum(nlp1v_pad[:, :2], axis=-1)
                nlp1v_pad = np.concatenate(
                    [nlp1v_pad, add_col.reshape(-1, 1)], axis=-1
                )
                nlp1v_pad[len_valid + 1:, :] = -1
                nlp1v_pad[:len_valid, :2] = 1 * (nlp1v_pad[:len_valid, :2] > 0)

                f_pos_pad_onehot2 = nlp1_pad[:, :2].copy()
                f_pos_pad_onehot2[:len_valid, :] = 1 * (
                    f_pos_pad_onehot2[:len_valid, :] > 0
                )
                f_pos_pad_onehot2[len_valid, :] = -1

                lst_trialinfo.append(values)
                lst_choice.append(eye2[choice])
                lst_rt.append(float(s['rt'][j]))
                lst_fsmr.append(fsmr)
                lst_pad_posdu.append(arr_posdu_pad)
                lst_pad_posduchoice.append(arr_posduchoice_pad)
                lst_pad_fpos_onehot2.append(f_pos_pad_onehot2)
                lst_pad_nlp1.append(nlp1_pad)
                lst_pad_nlp2.append(nlp2_pad)
                lst_pad_nlp1v.append(nlp1v_pad)

            fe_subj['trialinfo'] = np.asarray(lst_trialinfo, dtype=float)
            fe_subj['choice'] = np.asarray(lst_choice, dtype=float)
            fe_subj['rt'] = np.asarray(lst_rt, dtype=float)
            fe_subj['fsmr'] = np.asarray(lst_fsmr, dtype=float)
            fe_subj['posdu'] = np.asarray(lst_pad_posdu, dtype=np.int32)
            fe_subj['posduchoice'] = np.asarray(lst_pad_posduchoice, dtype=np.float32)
            fe_subj['fpos_onehot2'] = np.asarray(lst_pad_fpos_onehot2, dtype=np.float32)
            fe_subj['nlp1'] = np.asarray(lst_pad_nlp1, dtype=np.float32)
            fe_subj['nlp2'] = np.asarray(lst_pad_nlp2, dtype=np.float32)
            fe_subj['nlp1v'] = np.asarray(lst_pad_nlp1v, dtype=np.float32)
            fe_subj['params'] = s['params']

            if fe_subj['trialinfo'].shape[0] > 5:
                fe.append(fe_subj)

        with open(f'{dst}/{g}_batch0.pkl', 'wb') as f:
            pickle.dump(fe, f, protocol=4)
        print(f'saved {dst}/{g}_batch0.pkl')

    print(f'{m}: skipped trials exceeding fixed maxT: {n_skipped}')
