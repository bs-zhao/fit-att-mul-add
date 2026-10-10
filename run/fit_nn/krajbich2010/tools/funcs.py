import ast
from itertools import groupby

import numpy as np
import pandas as pd


def tp2arr(lst):
    values, counts = [], []
    for key, group in groupby(lst):
        values.append(key)
        counts.append(len(list(group)))
    return np.asarray(values), np.asarray(counts)


def arr2tp(values, counts):
    return np.array([
        val for val, count in zip(values, counts)
        for _ in range(int(count))
    ])


def _parse_array(x, dtype=int):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=dtype)
    if pd.isna(x):
        return np.asarray([], dtype=dtype)
    s = str(x).strip()
    try:
        return np.asarray(ast.literal_eval(s), dtype=dtype)
    except Exception:
        s = s.strip('[]')
        if not s:
            return np.asarray([], dtype=dtype)
        return np.fromstring(s.replace(',', ' '), sep=' ', dtype=dtype)


# Retain original RT as separate columns; 'rt' is fixation-only.
def load_food_data(path='../../../data/krajbich2010/trial_eye.csv'):
    df = pd.read_csv(path)
    req = ['subj', 'v0', 'v1', 'response', 'arr_ml3_left', 'arr_ml3_time']
    miss = [x for x in req if x not in df.columns]
    if miss:
        raise ValueError(f'missing required columns: {miss}')

    df = df.copy()
    df['arr_pos'] = df['arr_ml3_left'].apply(lambda x: _parse_array(x, int))
    df['arr_du'] = df['arr_ml3_time'].apply(lambda x: _parse_array(x, int))
    df = df[df.apply(
        lambda r: len(r.arr_pos) > 0 and len(r.arr_pos) == len(r.arr_du),
        axis=1,
    )].copy()
    df['choice'] = df['response'].astype(int)
    df['rt_tp'] = df['arr_du'].apply(lambda x: int(np.sum(x)))
    df['rt'] = df['rt_tp'] * 0.001  # Scheme A: summed fixation duration
    if 'rt_original_ms' in df.columns:
        df['rt_original_s'] = df['rt_original_ms'] * 0.001
    if 'rt_fixations_ms' in df.columns:
        check = df['rt_tp'].to_numpy() == df['rt_fixations_ms'].to_numpy()
        if not check.all():
            raise ValueError('Fixation sums differ from s0 preprocessing output')
    return df


def get_summary(arr, delta_t):
    arr = np.asarray(arr, dtype=int)
    eye = arr[(arr == 0) | (arr == 1)]
    if len(eye) == 0:
        return [0.0] * 12

    pos, du = tp2arr(eye)
    rt = max(len(eye) * delta_t, 1e-9)
    f = {}
    f['ttft0'] = np.sum(eye == 0) * delta_t
    f['ttft1'] = np.sum(eye == 1) * delta_t
    f['relft0'] = f['ttft0'] / rt
    f['relft1'] = f['ttft1'] / rt
    f['numf'] = len(pos)
    f['numf0'] = np.sum(pos == 0)
    f['numf1'] = np.sum(pos == 1)
    fst = np.eye(2)[eye[0]]
    lst = np.eye(2)[eye[-1]]
    f['fstf0'], f['fstf1'] = fst
    f['lstf0'], f['lstf1'] = lst
    f['switches'] = max(len(pos) - 1, 0)
    return list(f.values())


def build_sequence_with_strength(arr, choice, v, broadcast_choice=True):
    """Binary-choice version of adm-sbi's fixed-length time representation.

    Columns 0:2 encode the currently fixated option using that option's value.
    Columns 2:4 encode the binary choice. Padding rows are -1, and the first
    row after the valid fixation sequence is a choice token.
    """
    arr = np.asarray(arr, dtype=int)
    v = np.asarray(v, dtype=float)
    length = len(arr)
    valid_mask = arr != -1
    valid_arr = arr[valid_mask]

    output = np.full((length, 4), -1.0, dtype=np.float32)
    if len(valid_arr) > 0:
        fixation_matrix = np.zeros((len(valid_arr), 4), dtype=np.float32)
        fixation_matrix[np.arange(len(valid_arr)), valid_arr] = v[valid_arr]
        if broadcast_choice:
            fixation_matrix[:, 2 + int(choice)] = 1.0
        output[:len(valid_arr)] = fixation_matrix

    if len(valid_arr) < length:
        choice_vec = np.zeros(4, dtype=np.float32)
        choice_vec[2 + int(choice)] = 1.0
        output[len(valid_arr)] = choice_vec

    return output


def make_params(info, rng=np.random):
    p = {}
    for i, n in enumerate(info['free_pnames']):
        lo, hi = info['free_pranges'][i]
        p[n] = float(rng.uniform(lo, hi))
    for n, v in zip(info['fixed_pnames'], info['fixed_pvalues']):
        p[n] = v
    return p
