"""Preflight audit for Krajbich 2010, before expensive simulation/training.

Run from repository root: python run/fit_nn/krajbich2010/preflight.py

No files are written. Checks Python syntax of all entrypoints, shared
prior ranges, input schema, Scheme-A fixation RT, maxT and dependencies.
"""
import ast
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from tools.funcs import load_food_data
from tools.maxT import MAX_LEN_NF, MAX_LEN_RT
from tools.model_info import model_infos

ALL_MODELS = [
    'aDDM_1', 'aDDM_2', 'aDDM_t', 'aDDM_g',
    'aRACE_1', 'aRACE_2', 'aRACE_t', 'aRACE_g',
]
DDM = ALL_MODELS[:4]
ACC = ALL_MODELS[4:]


def assert_equal_priors(names, expected):
    for name in names:
        info = model_infos[name]
        actual = dict(zip(info['free_pnames'], info['free_pranges']))
        for param, bounds in expected.items():
            assert tuple(actual[param]) == tuple(bounds), (name, param, actual[param])
        ml_actual = dict(zip(info['free_pnames_ml'], info['free_pranges_ml']))
        for param, bounds in expected.items():
            assert tuple(ml_actual[param]) == tuple(bounds), (name, param, ml_actual[param])


def main():
    scripts = [
        's1_gen1w.py', 's2_getmaxRT.py', 's3_fe1.py',
        's4_train_range_huber.py', 's5_fe_real_data.py',
        's5_fe_real_data_maxT.py', 's5_fe_test_data_maxT.py',
        'tools/funcs.py', 'tools/model_info.py', 'tools/maxT.py',
        'compare_regen/s1_regen.py', 'compare_regen/s2_getmaxRT.py',
        'compare_regen/s3_fe1_maxT.py',
        'compare_regen/s4_train_classifier_maxT.py',
        'compare_regen/s6_decode_maxT.py',
        'compare_regen/s6_decode_maxT_mr.py',
    ]
    for script in scripts:
        file = HERE / script
        assert file.is_file(), f'Missing script: {file}'
        ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
        if 'tools/' not in script:
            source = file.read_text(encoding='utf-8')
            assert 'outputs/food_equal' not in source, (script, 'legacy output path')
            assert 'data/food_equal/' not in source, (script, 'legacy input path')
    assert set(model_infos) == set(ALL_MODELS)
    assert_equal_priors(DDM, {'d': (0.06, 2.8), 'a': (2.3, 10.5)})
    assert_equal_priors(ACC, {'d': (0.035, 2.3), 'a': (2.0, 14.5)})
    assert MAX_LEN_RT == 28287 and MAX_LEN_NF == 128
    for p in ['c_bifood', 'data2param_flow', 'data2param']:
        found = importlib.util.find_spec(p) is not None
        print(f'dependency {p}:', 'OK' if found else 'MISSING')
        if not found:
            print(f'  Install or activate the same environment used by food_equal for {p}')

    data_path = ROOT / 'outputs/krajbich2010/s0_prepare_data/trial_eye.csv'
    assert data_path.is_file(), (
        f'Missing {data_path}; run python run/test_krajbich2010/s0_prepare_data.py'
    )
    # The loader must receive an absolute path here because it intentionally
    # preserves food_equal's original relative-path interface.
    df = load_food_data(str(data_path))
    needed = {'rt_original_ms', 'rt_original_s', 'rt_fixations_ms',
              'subj', 'choice', 'arr_pos', 'arr_du', 'rt'}
    assert needed.issubset(df.columns), sorted(needed - set(df.columns))
    assert (df['rt_original_ms'] >= df['rt_fixations_ms']).all()
    assert np.allclose(df['rt'].to_numpy(), df['rt_fixations_ms'].to_numpy() / 1000.0)
    assert np.allclose(df['rt_original_s'].to_numpy(), df['rt_original_ms'].to_numpy() / 1000.0)
    assert df['choice'].isin([0, 1]).all()
    assert all(set(p).issubset({0, 1}) for p in df['arr_pos'])
    assert math.ceil(df['rt_original_ms'].max() * 1.20) == MAX_LEN_RT
    print(f'Python syntax: PASS ({len(scripts)} scripts)')
    print('Architecture-shared parameter priors: PASS')
    print('Scheme-A RT and preserved experimental RT: PASS')
    print('maxT: PASS', MAX_LEN_RT, 'ms')
    print('Trial count:', len(df), '; subjects:', df.subj.nunique())
    print('Preflight complete. Does NOT run SBI or write output files.')


if __name__ == '__main__':
    main()
