"""Validate shared DDM/ACC prior bounds across more Krajbich 2010 subjects.

Wraps the existing S4 simulator, keeping its shared-within-architecture
candidate prior definitions and original model equations. Then evaluates
extreme RTs, timeout-prone parameter draws, and subject heterogeneity.

From repo root:
  python run/test_krajbich2010/s5_validate_shared_priors.py --n-subjects 12 --n-draws 12 --n-trials 20
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S4 = Path(__file__).with_name('s4_calibrate_shared_priors.py')
DEFAULT_OUTPUT = ROOT / 'outputs/krajbich2010/s5_validate_shared_priors'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--n-subjects', type=int, default=12, help='Set 39 for all subjects')
    ap.add_argument('--n-draws', type=int, default=12, help='Parameter draws per model and subject')
    ap.add_argument('--n-trials', type=int, default=20)
    ap.add_argument('--candidate', nargs='+', default=['s3_shared', 'wide_1', 'wide_2'])
    ap.add_argument('--candidate-config', type=Path, default=None)
    ap.add_argument('--rt-margin', type=float, default=.20)
    ap.add_argument('--fast-rt-ms', type=float, default=300.)
    ap.add_argument('--max-fast-fraction', type=float, default=.05)
    ap.add_argument('--max-timeout-fraction', type=float, default=.10)
    ap.add_argument('--max-pathological-draw-fraction', type=float, default=.15)
    ap.add_argument('--min-coverage-fraction', type=float, default=.20)
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--exclude-subjects', nargs='*', type=float, default=[])
    args = ap.parse_args()

    if min(args.n_subjects, args.n_draws, args.n_trials) < 1:
        ap.error('subject, draw and trial counts must be positive')
    if args.rt_margin < 0 or args.fast_rt_ms <= 0:
        ap.error('RT margin must be nonnegative and fast RT threshold positive')
    for name in ('max_fast_fraction', 'max_timeout_fraction',
                 'max_pathological_draw_fraction', 'min_coverage_fraction'):
        if not 0 <= getattr(args, name) <= 1:
            ap.error(f'{name} must be in [0,1]')

    # All simulations continue to use S4's SAME parameter rectangles, SAME
    # source trials for all models, and SAME simulator; no code duplication.
    raw_out = args.output / 's4_raw'
    command = [
        sys.executable, str(S4),
        '--output', str(raw_out),
        '--candidate', *args.candidate,
        '--n-subjects', str(args.n_subjects),
        '--n-draws', str(args.n_draws),
        '--n-trials', str(args.n_trials),
        '--rt-margin', str(args.rt_margin),
        '--fast-rt-ms', str(args.fast_rt_ms),
        '--seed', str(args.seed),
    ]
    if args.candidate_config is not None:
        command.extend(['--candidate-config', str(args.candidate_config.resolve())])
    if args.exclude_subjects:
        command.extend(['--exclude-subjects', *map(str, args.exclude_subjects)])
    print('Running prior predictive simulation:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)

    draws = pd.read_csv(raw_out / 'draw_diagnostics.csv')
    sim_meta = json.loads((raw_out / 'calibration_report.json').read_text())
    if draws.empty:
        raise ValueError('S4 generated no parameter draws')

    # Count pathological draws separately. A 10% overall timeout rate can
    # conceal some parameter settings where most trials fail to respond.
    draws['pathological_draw'] = draws.n_timeout >= draws.n_sim.map(
        lambda n: math.ceil(n * .25))
    draws['covered_recorded_strict'] = (
        (draws.hit_rate >= .90) &
        draws.rt_ratio_recorded.between(.75, 1.25)
    )

    summary = []
    for (arch, candidate, model), g in draws.groupby(
            ['architecture', 'candidate', 'model'], sort=False):
        n = int(g.n_sim.sum())
        summary.append({
            'architecture': arch, 'candidate': candidate, 'model': model,
            'simulated_trials': n,
            'fast_rate': float(g.n_fast.sum() / n),
            'timeout_rate': float(g.n_timeout.sum() / n),
            'pathological_draw_fraction': float(g.pathological_draw.mean()),
            'recorded_rt_coverage_fraction': float(g.draw_coverage_recorded.mean()),
            'recorded_rt_strict_coverage_fraction': float(g.covered_recorded_strict.mean()),
            'median_rt_ratio_recorded': float(g.rt_ratio_recorded.median()),
            'median_rt_ratio_fixation': float(g.rt_ratio_fixation.median()),
            'fraction_any_timeout_draws': float((g.n_timeout > 0).mean()),
        })
    models = pd.DataFrame(summary)

    subjects = draws.groupby(
        ['architecture', 'candidate', 'model', 'source_subject'], sort=False
    ).agg(
        n_sim=('n_sim', 'sum'),
        n_fast=('n_fast', 'sum'),
        n_timeout=('n_timeout', 'sum'),
        pathological_draw_fraction=('pathological_draw', 'mean'),
        recorded_rt_coverage_fraction=('draw_coverage_recorded', 'mean'),
        median_rt_ratio_recorded=('rt_ratio_recorded', 'median')
    ).reset_index()
    subjects['fast_rate'] = subjects.n_fast / subjects.n_sim
    subjects['timeout_rate'] = subjects.n_timeout / subjects.n_sim

    # One shared d/a interval must pass criteria for ALL FOUR attention
    # branches within an architecture (not four hand-picked separate priors).
    ranking = models.groupby(['architecture', 'candidate'], sort=False).agg(
        worst_branch_fast_rate=('fast_rate', 'max'),
        worst_branch_timeout_rate=('timeout_rate', 'max'),
        worst_branch_pathological_fraction=('pathological_draw_fraction', 'max'),
        weakest_branch_coverage=('recorded_rt_coverage_fraction', 'min')
    ).reset_index()
    ranking['passes_screen'] = (
        (ranking.worst_branch_fast_rate <= args.max_fast_fraction) &
        (ranking.worst_branch_timeout_rate <= args.max_timeout_fraction) &
        (ranking.worst_branch_pathological_fraction <= args.max_pathological_draw_fraction) &
        (ranking.weakest_branch_coverage >= args.min_coverage_fraction)
    )
    candidates = sim_meta['candidates']
    ranking['rectangle_area'] = ranking.apply(
        lambda row: (
            candidates[row.architecture][row.candidate]['d'][1] -
            candidates[row.architecture][row.candidate]['d'][0]
        ) * (
            candidates[row.architecture][row.candidate]['a'][1] -
            candidates[row.architecture][row.candidate]['a'][0]
        ), axis=1
    )
    ranking = ranking.sort_values(
        ['architecture', 'passes_screen', 'rectangle_area'],
        ascending=[True, False, False]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    models.to_csv(args.output / 'model_summary.csv', index=False)
    subjects.to_csv(args.output / 'subject_summary.csv', index=False)
    ranking.to_csv(args.output / 'candidate_ranking.csv', index=False)

    report = {
        'source_data_max_recorded_rt_ms': sim_meta['observed_max_recorded_rt_ms'],
        'simulation_max_rt_ms': sim_meta['max_simulation_time_ms'],
        'subjects': sim_meta['sampling']['source_subjects'],
        'simulator_calls': sim_meta['sampling']['total_simulator_calls'],
        'shared_candidate_ranges': candidates,
        'criteria': {
            'maximum_fraction_faster_than_300ms': args.max_fast_fraction,
            'maximum_fraction_timeouts': args.max_timeout_fraction,
            'maximum_fraction_draws_with_at_least_25pct_timeouts':
                args.max_pathological_draw_fraction,
            'minimum_draw_fraction_covering_recorded_rt': args.min_coverage_fraction,
            'recorded_rt_coverage_definition':
                'hit_rate>=0.90 and simulated/recorded median RT ratio [0.5,2.0]',
        },
        'cautions': [
            'Thresholds are diagnostics, not established scientific cutoffs.',
            'Larger prior rectangle area does not automatically make a uniform prior preferable.',
            'The original RT differs from summed option-fixation time; fitting time convention is not yet resolved.',
            'Some draws share subjects and fixation sequences; standard binomial independence is not assumed.',
            'c_bifood uses libc rand, so S4 noise trajectories are not fully reproducible with NumPy seed.',
            'This wrapper does not modify any official model priors or train an SBI model.',
        ]
    }
    (args.output / 'validation_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    print('\nS5 candidate ranking:')
    print(ranking.to_string(index=False))
    print('Saved S5 summary to:', args.output)
    print('Saved raw simulator draws to:', raw_out)


if __name__ == '__main__':
    main()
