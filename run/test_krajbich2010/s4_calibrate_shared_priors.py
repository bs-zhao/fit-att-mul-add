"""Test shared within-architecture prior bounds, without changing official priors.

Uses the observed maximum *recorded* RT times (1 + --rt-margin) as the
simulation cutoff and proposed fixed-maxT length. Compares candidate d/a
RECTANGLES with the same bounds across all four branches of each
architecture. Tests both quick responses and no-hits, not just hit-only RT.

Outputs go only to outputs/krajbich2010/tests/s4_calibrate_shared_priors/.
This diagnostic does NOT fit a model or update model_info.py.

Run from repository root:
  python run/test_krajbich2010/s4_calibrate_shared_priors.py --n-subjects 4 --n-draws 12 --n-trials 12
"""

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'run/fit_nn/food_equal'))

from tools.model_info import model_infos  # noqa: E402

MODEL_GROUPS = {
    'DDM': ['aDDM_1', 'aDDM_2', 'aDDM_t', 'aDDM_g'],
    'ACC': ['aRACE_1', 'aRACE_2', 'aRACE_t', 'aRACE_g'],
}

# Candidate prior bounds (not model-specific or branch-specific). Change these
# only in this testing script until calibration is complete. Existing priors
# are included as a point of comparison, not adopted as defaults.
CANDIDATES = {
    'DDM': {
        'original': {'d': [0.1, 5.0], 'a': [1.0, 5.0]},
        's3_shared': {'d': [0.05, 2.5], 'a': [2.0, 10.0]},
        'wide_1': {'d': [0.03, 3.0], 'a': [1.5, 12.0]},
        'wide_2': {'d': [0.02, 4.0], 'a': [1.25, 15.0]},
    },
    'ACC': {
        'original': {'d': [0.1, 8.0], 'a': [0.5, 5.0]},
        's3_shared': {'d': [0.025, 2.0], 'a': [1.5, 15.0]},
        'wide_1': {'d': [0.015, 2.5], 'a': [1.5, 16.0]},
        'wide_2': {'d': [0.01, 3.0], 'a': [1.25, 18.0]},
    },
}


def load_source(path, seed, n_subjects, n_trials, max_time_ms, exclude_subjects=()):
    df = pd.read_csv(path)
    required = ['subj', 'trial', 'v0', 'v1', 'response',
                'arr_ml3_left', 'arr_ml3_time', 'rt_original_ms', 'rt_fixations_ms']
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns {missing}; rerun s0_prepare_data.py')
    rng = np.random.default_rng(seed)
    datasets = []
    eligible = []
    excluded = set(map(float, exclude_subjects))
    for subj, sub in df.groupby('subj', sort=False):
        if float(subj) in excluded:
            continue
        sub = sub.loc[(sub.rt_original_ms < max_time_ms) &
                      (sub.rt_fixations_ms < max_time_ms)].copy()
        if len(sub) >= n_trials:
            eligible.append((subj, sub))
    if len(eligible) < n_subjects:
        raise ValueError(f'Only {len(eligible)} subjects have {n_trials} valid trials')
    selected = rng.choice(len(eligible), size=n_subjects, replace=False)
    for idx in selected:
        subj, sub = eligible[int(idx)]
        selection = rng.choice(len(sub), size=n_trials, replace=False)
        trials = []
        for _, row in sub.iloc[selection].iterrows():
            pos = np.ascontiguousarray(ast.literal_eval(row.arr_ml3_left), dtype=np.int64)
            dur = np.ascontiguousarray(ast.literal_eval(row.arr_ml3_time), dtype=np.int64)
            if len(pos) < 1 or len(pos) != len(dur) or not (dur > 0).all():
                raise ValueError(f'Invalid fixation sequence: {subj}, {row.trial}')
            if not set(pos).issubset({0, 1}):
                raise ValueError(f'Invalid fixation side: {subj}, {row.trial}')
            trials.append({
                'values': np.ascontiguousarray([row.v0, row.v1], dtype=np.float64),
                'pos': pos, 'dur': dur,
                'recorded_rt_s': float(row.rt_original_ms) / 1000,
                'fixation_rt_s': float(row.rt_fixations_ms) / 1000,
            })
        datasets.append((subj, trials))
    return df, datasets


def instantiate_params(info, d, a, u_nuisance):
    params = dict(zip(info['fixed_pnames'], info['fixed_pvalues']))
    nuis_ix = 0
    for name, (low, high) in zip(info['free_pnames'], info['free_pranges']):
        if name == 'd':
            params[name] = float(d)
        elif name == 'a':
            params[name] = float(a)
        else:
            params[name] = float(low + u_nuisance[nuis_ix] * (high - low))
            nuis_ix += 1
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'data/krajbich2010/trial_eye.csv')
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/krajbich2010/tests/s4_calibrate_shared_priors')
    ap.add_argument('--architecture', nargs='+', choices=['DDM', 'ACC'], default=['DDM', 'ACC'])
    ap.add_argument('--candidate', nargs='+', default=['original', 's3_shared', 'wide_1', 'wide_2'])
    ap.add_argument('--candidate-config', type=Path, default=None,
                    help='Optional JSON with extra DDM/ACC candidate d/a rectangles')
    ap.add_argument('--rt-margin', type=float, default=0.20,
                    help='Simulation time limit = ceil(max observed recorded RT in ms * (1+margin))')
    ap.add_argument('--fast-rt-ms', type=float, default=300.)
    ap.add_argument('--max-fast-rate', type=float, default=0.10)
    ap.add_argument('--max-timeout-rate', type=float, default=0.10)
    ap.add_argument('--min-coverage-fraction', type=float, default=0.05)
    ap.add_argument('--n-subjects', type=int, default=4)
    ap.add_argument('--n-draws', type=int, default=12)
    ap.add_argument('--n-trials', type=int, default=12)
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--exclude-subjects', nargs='*', type=float, default=[],
                    help='Source subject IDs held out of this run')
    args = ap.parse_args()

    # Extend the diagnostic candidate catalog without modifying production priors.
    catalog = {arch: dict(items) for arch, items in CANDIDATES.items()}
    if args.candidate_config is not None:
        custom = json.loads(args.candidate_config.read_text(encoding='utf-8'))
        if not isinstance(custom, dict):
            ap.error('candidate-config must be a JSON object')
        for arch, items in custom.items():
            if arch not in catalog or not isinstance(items, dict):
                ap.error(f'Invalid candidate-config architecture: {arch}')
            for name, bounds in items.items():
                if not isinstance(bounds, dict) or set(bounds) != {'d', 'a'}:
                    ap.error(f'Invalid candidate bounds: {arch}/{name}')
                cleaned = {}
                for parameter in ('d', 'a'):
                    limits = bounds[parameter]
                    if (not isinstance(limits, list) or len(limits) != 2 or
                            not all(isinstance(v, (int, float)) and math.isfinite(v) for v in limits) or
                            limits[0] <= 0 or limits[0] >= limits[1]):
                        ap.error(f'Invalid {parameter} prior interval: {arch}/{name}')
                    cleaned[parameter] = [float(v) for v in limits]
                if name in catalog[arch]:
                    ap.error(f'Candidate name already exists: {arch}/{name}')
                catalog[arch][name] = cleaned
    for arch in args.architecture:
        unknown = [name for name in args.candidate if name not in catalog[arch]]
        if unknown:
            ap.error(f'Unknown candidates for {arch}: {unknown}')

    if not args.input.exists():
        raise FileNotFoundError(f'Run s0_prepare_data.py first; missing {args.input}')
    if min(args.n_subjects, args.n_draws, args.n_trials) < 1:
        ap.error('n-subjects, n-draws and n-trials must be positive')
    if args.rt_margin < 0:
        ap.error('rt-margin must not be negative')
    for field in ('max_fast_rate', 'max_timeout_rate', 'min_coverage_fraction'):
        if not (0 <= getattr(args, field) <= 1):
            ap.error(f'{field} must be in [0,1]')
    if args.fast_rt_ms <= 0:
        ap.error('fast-rt-ms must be positive')

    from c_bifood import sim_trial_aDDM, sim_trial_aRACE

    # Simulation MAX RT and eventual padded preprocessing length must agree.
    src = pd.read_csv(args.input, usecols=['rt_original_ms'])
    max_observed_rt_ms = float(src.rt_original_ms.max())
    max_time_ms = math.ceil(max_observed_rt_ms * (1.0 + args.rt_margin))
    max_time_s = max_time_ms / 1000.0

    df, datasets = load_source(args.input, args.seed, args.n_subjects, args.n_trials,
                               max_time_ms, exclude_subjects=args.exclude_subjects)
    rng = np.random.default_rng(args.seed + 11)
    uniforms_da = rng.random((args.n_subjects, args.n_draws, 2))
    uniforms_nuisance = rng.random((args.n_subjects, args.n_draws, 2))
    results = []
    for arch in args.architecture:
        sim = sim_trial_aDDM if arch == 'DDM' else sim_trial_aRACE
        for cand in args.candidate:
            bounds = catalog[arch][cand]
            for model in MODEL_GROUPS[arch]:
                info = model_infos[model]
                for subject_ix, (subj, trials) in enumerate(datasets):
                    true_rt = np.median([t['recorded_rt_s'] for t in trials])
                    fix_rt = np.median([t['fixation_rt_s'] for t in trials])
                    for draw in range(args.n_draws):
                        u_d, u_a = uniforms_da[subject_ix, draw]
                        d = bounds['d'][0] + u_d * (bounds['d'][1] - bounds['d'][0])
                        a = bounds['a'][0] + u_a * (bounds['a'][1] - bounds['a'][0])
                        params = instantiate_params(info, d, a, uniforms_nuisance[subject_ix, draw])
                        sim_rt = []
                        n_fast = n_hit = 0
                        for trial in trials:
                            (choice, rt, hit), _ = sim(
                                params,
                                trial['values'].copy(), trial['pos'].copy(), trial['dur'].copy(),
                                extend_last=500000, repeat=0, dt=.001, rcd=0,
                                max_rt=max_time_s,
                            )
                            if int(hit):
                                n_hit += 1
                                sim_rt.append(float(rt))
                                if rt * 1000.0 < args.fast_rt_ms:
                                    n_fast += 1
                        n_sim = len(trials)
                        median_rt = float(np.median(sim_rt)) if sim_rt else np.nan
                        hit_rate = n_hit / n_sim
                        results.append({
                            'architecture': arch, 'candidate': cand, 'model': model,
                            'source_subject': subj, 'draw': draw,
                            'd': d, 'a': a, 'n_sim': n_sim, 'n_hit': n_hit,
                            'n_timeout': n_sim - n_hit, 'n_fast': n_fast,
                            'hit_rate': hit_rate,
                            'sim_hit_median_rt_s': median_rt,
                            'source_recorded_median_rt_s': float(true_rt),
                            'source_fixation_median_rt_s': float(fix_rt),
                            'rt_ratio_recorded': median_rt / true_rt,
                            'rt_ratio_fixation': median_rt / fix_rt,
                            'draw_coverage_recorded': bool(hit_rate >= .90 and
                                                           np.isfinite(median_rt) and
                                                           .5 <= median_rt / true_rt <= 2),
                        })
                print(f'completed {arch} / {cand} / {model}', flush=True)

    draws = pd.DataFrame(results)
    aggregate = []
    for (arch, candidate, model), group in draws.groupby(['architecture', 'candidate', 'model'], sort=False):
        bounds = catalog[arch][candidate]
        n = int(group.n_sim.sum())
        aggregate.append({
            'architecture': arch, 'candidate': candidate, 'model': model,
            'd_low': bounds['d'][0], 'd_high': bounds['d'][1],
            'a_low': bounds['a'][0], 'a_high': bounds['a'][1],
            'rectangle_area': (bounds['d'][1] - bounds['d'][0]) * (bounds['a'][1] - bounds['a'][0]),
            'n_simulations': n,
            'fraction_rt_lt_fast_threshold': float(group.n_fast.sum() / n),
            'fraction_timeout': float(group.n_timeout.sum() / n),
            'median_hit_rt_s': float(group.sim_hit_median_rt_s.median()),
            'median_rt_ratio_recorded': float(group.rt_ratio_recorded.median()),
            'median_rt_ratio_fixation': float(group.rt_ratio_fixation.median()),
            'fraction_parameter_draws_cover_recorded': float(group.draw_coverage_recorded.mean()),
            'fraction_draws_with_timeout': float((group.n_timeout > 0).mean()),
        })
    per_model = pd.DataFrame(aggregate)
    ranking = per_model.groupby(['architecture', 'candidate'], sort=False).agg(
        rectangle_area=('rectangle_area', 'first'),
        worst_model_fast_fraction=('fraction_rt_lt_fast_threshold', 'max'),
        worst_model_timeout_fraction=('fraction_timeout', 'max'),
        weakest_model_coverage_fraction=('fraction_parameter_draws_cover_recorded', 'min'),
    ).reset_index()
    ranking['passes_screen'] = (
        (ranking.worst_model_fast_fraction <= args.max_fast_rate) &
        (ranking.worst_model_timeout_fraction <= args.max_timeout_rate) &
        (ranking.weakest_model_coverage_fraction >= args.min_coverage_fraction)
    )
    ranking = ranking.sort_values(['architecture', 'passes_screen', 'rectangle_area'],
                                  ascending=[True, False, False])

    args.output.mkdir(parents=True, exist_ok=True)
    draws.to_csv(args.output / 'draw_diagnostics.csv', index=False)
    per_model.to_csv(args.output / 'model_candidate_summary.csv', index=False)
    ranking.to_csv(args.output / 'candidate_ranking.csv', index=False)
    report = {
        'observed_max_recorded_rt_ms': max_observed_rt_ms,
        'rt_margin': args.rt_margin,
        'excluded_source_subjects': args.exclude_subjects,
        'max_simulation_time_ms': max_time_ms,
        'max_simulation_time_s': max_time_s,
        'proposed_MAX_LEN_RT_ms': max_time_ms,
        'proposed_MAX_LEN_NF': 128,
        'source_trial_count': len(df),
        'sampling': {'source_subjects': [float(s) for s, _ in datasets],
                     'n_draws': args.n_draws,
                     'n_trials_per_subject': args.n_trials,
                     'total_simulator_calls': int(draws.n_sim.sum())},
        'screen_thresholds': {
            'fast_rt_ms': args.fast_rt_ms,
            'max_fast_fraction': args.max_fast_rate,
            'max_timeout_fraction': args.max_timeout_rate,
            'min_fraction_draws_covered': args.min_coverage_fraction,
            'covered_draw_definition': 'hit_rate>=0.90 and simulated/recorded RT median ratio in [0.5,2.0]',
        },
        'candidates': {arch: {cand: catalog[arch][cand] for cand in args.candidate}
                       for arch in args.architecture},
        'important': [
            'Only shared within-architecture d/a ranges; identical across all four gaze branches.',
            'Theta and gamma retain the original branch-specific ranges where freely estimated.',
            'Candidate rectangle ranking is a prior-predictive diagnostic, NOT a fitted or recommended model.',
            'Non-hits are counted; hit-only RT distributions are selection-biased.',
            'Original RT and fixation-duration sum differ systematically. No missing time is imputed.',
            'Simulation uses C libc rand; NumPy random seed alone does not fully fix trajectories.',
            'Do not alter formal model priors automatically or compare fitted-model evidences yet.',
        ],
    }
    with (args.output / 'calibration_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'Observed max RT={max_observed_rt_ms:.1f} ms; margin={args.rt_margin:.0%}; '
          f'max simulation time={max_time_s:.3f} s')
    print(ranking.to_string(index=False))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
