"""Small prior-predictive smoke test for all eight existing c_bifood models.

Requires installed c_bifood (same as the main food_equal pipeline).
Run from any directory:
    python run/test_krajbich2010/s2_prior_predictive.py

This is NOT model fitting. It samples parameters from the unchanged current
priors, runs simulated choices with the Krajbich fixation input/value range,
and measures hitting probability and RT coverage.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FOOD_RUN = ROOT / 'run/fit_nn/food_equal'
sys.path.insert(0, str(FOOD_RUN))

from tools.funcs import make_params  # noqa: E402
from tools.model_info import model_infos  # noqa: E402

MODEL_NAMES = [
    'aDDM_1', 'aDDM_2', 'aDDM_t', 'aDDM_g',
    'aRACE_1', 'aRACE_2', 'aRACE_t', 'aRACE_g',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=ROOT / 'outputs/krajbich2010/s0_prepare_data/trial_eye.csv')
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/krajbich2010/s2_prior_predictive')
    ap.add_argument('--n-subjects', type=int, default=5, help='Synthetic subjects per model (one prior draw each)')
    ap.add_argument('--n-trials', type=int, default=20, help='Real fixation sequences sampled per synthetic subject')
    ap.add_argument('--seed', type=int, default=1729)
    args = ap.parse_args()
    if args.n_subjects < 1 or args.n_trials < 1:
        ap.error('n-subjects and n-trials must both be positive')
    if not args.input.exists():
        raise FileNotFoundError(f'Missing {args.input}; run s0_prepare_data.py first.')

    try:
        from c_bifood import sim_trial_aDDM, sim_trial_aRACE
    except ImportError as exc:
        raise RuntimeError('c_bifood must be installed in the SAME Python environment as the main fit_nn pipeline') from exc

    df = pd.read_csv(args.input)
    needed = ['subj', 'v0', 'v1', 'arr_ml3_left', 'arr_ml3_time', 'rt_original_ms']
    missing = [k for k in needed if k not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}; rerun s0_prepare_data.py')
    df = df.copy()
    df['pos'] = df.arr_ml3_left.map(ast.literal_eval)
    df['dur'] = df.arr_ml3_time.map(ast.literal_eval)
    if (df[['v0', 'v1']].min().min() < 0 or df[['v0', 'v1']].max().max() > 10):
        print('WARNING: observed food rating outside the expected 0-10 range')

    rng = np.random.default_rng(args.seed)
    subs = list(df.subj.unique())
    out = []
    for model in MODEL_NAMES:
        info = model_infos[model]
        sim = sim_trial_aDDM if model.startswith('aDDM') else sim_trial_aRACE
        for synthetic_subject in range(args.n_subjects):
            subj = subs[int(rng.integers(len(subs)))]
            trials = df.loc[df.subj == subj]
            params = make_params(info, rng=rng)
            selection = rng.choice(len(trials), size=args.n_trials, replace=args.n_trials > len(trials))
            for j in selection:
                row = trials.iloc[int(j)]
                positions = np.ascontiguousarray(np.asarray(row.pos, dtype=np.int64))
                durations = np.ascontiguousarray(np.asarray(row.dur, dtype=np.int64))
                values = np.ascontiguousarray(np.asarray([row.v0, row.v1], dtype=np.float64))
                fixation_rt_s = float(durations.sum()) / 1000.0
                if len(positions) == 0 or len(positions) != len(durations):
                    raise ValueError(f'Invalid fixation sequence at subj {subj}')
                # Exactly match s1_gen1w.py: extend final fixation after its recorded duration.
                # Always COPY inputs because c_bifood may mutate the supplied final duration.
                (choice, rt, hit), _ = sim(params, values, positions.copy(), durations.copy(),
                                          extend_last=500000, repeat=0,
                                          dt=0.001, rcd=0, max_rt=14.5)
                out.append({
                    'model': model,
                    'synthetic_subject': synthetic_subject,
                    'source_subject': subj,
                    'source_trial': row.trial,
                    'v0_right': row.v0,
                    'v1_left': row.v1,
                    'choice_left': int(choice),
                    'hit': int(hit),
                    'rt_sim_s': float(rt),
                    'rt_source_s': float(row.rt_original_ms / 1000),
                    'rt_fixations_s': fixation_rt_s,
                })
        print('finished', model, flush=True)

    simulated = pd.DataFrame(out)
    summaries = []
    for model, g in simulated.groupby('model', sort=False):
        hit_rt = g.loc[g.hit == 1, 'rt_sim_s']
        summaries.append({
            'model': model,
            'n_simulated_trials': len(g),
            'n_hits': len(hit_rt),
            'hit_rate': float(g.hit.mean()),
            'sim_rt_hit_median_s': float(hit_rt.median()) if len(hit_rt) else np.nan,
            'sim_rt_hit_p95_s': float(hit_rt.quantile(.95)) if len(hit_rt) else np.nan,
            'sampled_source_rt_median_s': float(g.rt_source_s.median()),
            'sampled_fixation_rt_median_s': float(g.rt_fixations_s.median()),
        })
    summary = pd.DataFrame(summaries)
    report = {
        'n_subjects_per_model': args.n_subjects,
        'n_trials_per_synthetic_subject': args.n_trials,
        'random_seed_python_numpy': args.seed,
        'important': 'Cython c_bifood uses libc rand; seed here controls trial/parameter sampling but '
                     'does not guarantee identical diffusion trajectories across runs.',
        'interpretation': 'Prior predictive quick check only, not a posterior fit or model ranking. '
                          'A low hit rate or severe RT mismatch indicates prior-range/simulation compatibility issues.',
        'model_results': summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    simulated.to_csv(args.output / 'simulated_trials.csv', index=False)
    summary.to_csv(args.output / 'model_summary.csv', index=False)
    with (args.output / 'prior_predictive_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(summary.to_string(index=False))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
