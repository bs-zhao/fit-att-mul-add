"""Diagnostic prior coverage sweep; DOES NOT change training priors or model equations.

All models see the same participant/trial subsamples. For each parameter draw,
test the original d/a prior as well as diagnostic drift/boundary scalings.
Compare simulated hit RT against both recorded RT and fixation-duration RT.
Use this as a screening tool before any full SBI training.

From repository root:
    python run/test_krajbich2010/s3_prior_coverage.py
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIT_DIR = ROOT / 'run/fit_nn/food_equal'
sys.path.insert(0, str(FIT_DIR))

from tools.funcs import make_params  # noqa: E402
from tools.model_info import model_infos  # noqa: E402

MODEL_NAMES = [
    'aDDM_1', 'aDDM_2', 'aDDM_t', 'aDDM_g',
    'aRACE_1', 'aRACE_2', 'aRACE_t', 'aRACE_g',
]


def make_trials(rowset):
    trials = []
    for _, row in rowset.iterrows():
        pos = np.ascontiguousarray(np.array(ast.literal_eval(row.arr_ml3_left), dtype=np.int64))
        dur = np.ascontiguousarray(np.array(ast.literal_eval(row.arr_ml3_time), dtype=np.int64))
        if len(pos) == 0 or len(pos) != len(dur) or any(x <= 0 for x in dur):
            raise ValueError('Invalid fixation sequence')
        trials.append({
            'values': np.ascontiguousarray([row.v0, row.v1], dtype=np.float64),
            'pos': pos,
            'dur': dur,
            'choice': int(row.response),
            'rt_original': float(row.rt_original_ms) / 1000.0,
            'rt_fixations': float(row.rt_fixations_ms) / 1000.0,
        })
    return trials


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'data/krajbich2010/trial_eye.csv')
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/krajbich2010/tests/s3_prior_coverage')
    ap.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES,
                    help='Optional model subset for a targeted diagnostic sweep')
    ap.add_argument('--n-source-subjects', type=int, default=4)
    ap.add_argument('--n-draws', type=int, default=8)
    ap.add_argument('--n-trials', type=int, default=24)
    ap.add_argument('--d-scales', type=float, nargs='+', default=[1., .5, .25])
    ap.add_argument('--a-scales', type=float, nargs='+', default=[1., 2.])
    ap.add_argument('--seed', type=int, default=1729)
    args = ap.parse_args()
    if min(args.n_source_subjects, args.n_draws, args.n_trials) < 1:
        ap.error('counts must be positive')
    if any(x <= 0 for x in args.d_scales + args.a_scales):
        ap.error('parameter scales must be positive')

    try:
        from c_bifood import sim_trial_aDDM, sim_trial_aRACE
    except ImportError as e:
        raise RuntimeError('Install c_bifood in this Python environment before running.') from e

    df = pd.read_csv(args.input)
    required = ['subj', 'v0', 'v1', 'response', 'arr_ml3_left', 'arr_ml3_time',
                'rt_original_ms', 'rt_fixations_ms']
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f'Missing {missing}; rerun s0_prepare_data.py')

    # Avoid maxT exclusions in the benchmark and use the identical trials for all models.
    df['nf'] = df.arr_ml3_left.map(lambda x: len(ast.literal_eval(x)))
    df = df.loc[(df.rt_original_ms < 15000) &
                (df.rt_fixations_ms <= 14997) &
                (df.nf < 128)].copy()

    eligible = [subj for subj, group in df.groupby('subj') if len(group) >= args.n_trials]
    if len(eligible) < args.n_source_subjects:
        raise ValueError(f'Only {len(eligible)} subjects have {args.n_trials} eligible trials')

    rng = np.random.default_rng(args.seed)
    selected_subjects = rng.choice(np.array(eligible), size=args.n_source_subjects, replace=False)
    datasets = []
    for subj in selected_subjects:
        group = df.loc[df.subj == subj]
        rows = group.iloc[rng.choice(len(group), size=args.n_trials, replace=False)]
        datasets.append((subj, make_trials(rows)))

    rows = []
    for model in args.models:
        sim = sim_trial_aDDM if model.startswith('aDDM') else sim_trial_aRACE
        for subj, trials in datasets:
            target_original = float(np.median([t['rt_original'] for t in trials]))
            target_fixations = float(np.median([t['rt_fixations'] for t in trials]))
            for draw in range(args.n_draws):
                baseline_params = make_params(model_infos[model], rng=rng)
                for ds in args.d_scales:
                    for a_s in args.a_scales:
                        params = dict(baseline_params)
                        params['d'] = baseline_params['d'] * ds
                        params['a'] = baseline_params['a'] * a_s
                        hit_rt = []
                        agreed = []
                        for trial in trials:
                            (choice, rt, hit), _ = sim(
                                params, trial['values'].copy(), trial['pos'].copy(), trial['dur'].copy(),
                                extend_last=500000, repeat=0, dt=.001, rcd=0, max_rt=14.5)
                            if int(hit):
                                hit_rt.append(float(rt))
                                agreed.append(int(choice) == trial['choice'])
                        hit_rate = len(hit_rt) / len(trials)
                        rt_median = float(np.median(hit_rt)) if hit_rt else np.nan
                        rows.append({
                            'model': model,
                            'source_subject': subj,
                            'draw': draw,
                            'd_scale': ds,
                            'a_scale': a_s,
                            'd': params['d'],
                            'a': params['a'],
                            'n_trials': len(trials),
                            'hit_rate': hit_rate,
                            'sim_rt_median_s': rt_median,
                            'target_original_rt_median_s': target_original,
                            'target_fixation_rt_median_s': target_fixations,
                            'sim_to_original_ratio': rt_median / target_original,
                            'sim_to_fixation_ratio': rt_median / target_fixations,
                            'choice_agreement_given_hit': float(np.mean(agreed)) if agreed else np.nan,
                        })
        print('completed', model, flush=True)

    draws = pd.DataFrame(rows)
    valid = draws.hit_rate >= .90
    draws['original_25pct'] = valid & draws.sim_to_original_ratio.between(.75, 1.25)
    draws['fixation_25pct'] = valid & draws.sim_to_fixation_ratio.between(.75, 1.25)
    summary = draws.groupby(['model', 'd_scale', 'a_scale'], sort=False).agg(
        n_draws=('draw', 'size'),
        median_hit_rate=('hit_rate', 'median'),
        median_sim_rt_s=('sim_rt_median_s', 'median'),
        median_original_ratio=('sim_to_original_ratio', 'median'),
        median_fixation_ratio=('sim_to_fixation_ratio', 'median'),
        fraction_cover_original_25pct=('original_25pct', 'mean'),
        fraction_cover_fixation_25pct=('fixation_25pct', 'mean'),
        mean_choice_agreement=('choice_agreement_given_hit', 'mean'),
    ).reset_index()

    args.output.mkdir(parents=True, exist_ok=True)
    draws.to_csv(args.output / 'parameter_draws.csv', index=False)
    summary.to_csv(args.output / 'coverage_summary.csv', index=False)
    report = {
        'source_subjects': [float(s) for s in selected_subjects],
        'models': args.models,
        'draws_per_subject_model': args.n_draws,
        'trials_per_subject': args.n_trials,
        'n_total_simulations': int(len(draws) * args.n_trials),
        'd_scales': args.d_scales,
        'a_scales': args.a_scales,
        'rt_calibration_window': 'simulated median / target median between 0.75 and 1.25, and hit_rate >= 0.90',
        'important': 'Screening only: random-prior diagnostic, not individual-level model fitting. '
                     'Non-baseline d/a scales can exceed training priors. Changing training priors '
                     'requires new simulations and SBI training. Simulations use libc rand, '
                     'so NumPy seed alone does not fully determine results.',
    }
    with (args.output / 'coverage_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(summary.to_string(index=False))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
