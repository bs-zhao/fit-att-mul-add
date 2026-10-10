"""Compare recorded RT with summed option-fixation durations; do not alter either.

Run from any directory:
    python run/test_krajbich2010/s1_rt_sensitivity.py
Inputs: outputs/krajbich2010/s0_prepare_data/trial_eye.csv
Outputs: outputs/krajbich2010/s1_rt_sensitivity/
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'outputs/krajbich2010/s0_prepare_data/trial_eye.csv'
DEFAULT_OUTPUT = ROOT / 'outputs/krajbich2010/s1_rt_sensitivity'


def within_subject_slopes(df, rt_col):
    """Descriptive within-subject OLS: log(RT) ~ |value difference| + value sum."""
    x = df[['value_diff_abs', 'value_sum']].copy()
    y = np.log(df[rt_col].to_numpy(dtype=float) / 1000.0)
    groups = df['subj']
    x = x - x.groupby(groups).transform('mean')
    y = y - pd.Series(y, index=df.index).groupby(groups).transform('mean').to_numpy()
    return np.linalg.lstsq(x.to_numpy(), y, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f'{args.input}; run s0_prepare_data.py first')

    df = pd.read_csv(args.input)
    required = ['subj', 'trial', 'v0', 'v1', 'response', 'arr_ml3_left',
                'arr_ml3_time', 'rt_original_ms', 'rt_fixations_ms']
    missing = [k for k in required if k not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}. Rerun updated s0_prepare_data.py.')

    positions = df['arr_ml3_left'].map(ast.literal_eval)
    durations = df['arr_ml3_time'].map(ast.literal_eval)
    check = np.array([len(p) == len(d) and len(p) > 0 for p, d in zip(positions, durations)])
    if not check.all():
        raise ValueError(f'{int((~check).sum())} trials have mismatched fixation position/duration lengths')
    calculated = np.array([sum(map(int, d)) for d in durations])
    if not np.array_equal(calculated, df['rt_fixations_ms'].to_numpy(dtype=int)):
        raise ValueError('Fixation sums are inconsistent with s0 RT column')
    if df[['rt_original_ms', 'rt_fixations_ms']].le(0).any().any():
        raise ValueError('Nonpositive RT')

    df['n_fixations'] = positions.map(len)
    df['gap_ms'] = df.rt_original_ms - df.rt_fixations_ms
    df['value_sum'] = df.v0 + df.v1
    df['value_diff_abs'] = (df.v1 - df.v0).abs()
    df['logrt_original'] = np.log(df.rt_original_ms / 1000.)
    df['logrt_fixations'] = np.log(df.rt_fixations_ms / 1000.)
    df['logrt_delta'] = df.logrt_original - df.logrt_fixations

    # The standard maxT implementation excludes ceil(fixation_points/3) >= 5000.
    ds, max_rt_ms, max_nf = 3, 15000, 128
    df['excluded_using_fixation_rt'] = (np.ceil(df.rt_fixations_ms / ds) >=
                                        np.floor(max_rt_ms / ds)) | (df.n_fixations >= max_nf)
    df['excluded_using_recorded_rt'] = (df.rt_original_ms >= max_rt_ms) | (df.n_fixations >= max_nf)
    slopes = {}
    for label, col in [('original', 'rt_original_ms'), ('fixations', 'rt_fixations_ms')]:
        x = within_subject_slopes(df, col)
        slopes[label] = {'abs_value_diff': float(x[0]), 'value_sum': float(x[1])}

    per_subject = df.groupby('subj', sort=False).agg(
        n_trials=('trial', 'size'),
        mean_gap_ms=('gap_ms', 'mean'),
        median_gap_ms=('gap_ms', 'median'),
        mean_original_rt_ms=('rt_original_ms', 'mean'),
        mean_fixation_rt_ms=('rt_fixations_ms', 'mean'),
        excluded_fixation=('excluded_using_fixation_rt', 'sum'),
        excluded_recorded=('excluded_using_recorded_rt', 'sum'),
    ).reset_index()

    report = {
        'subjects': int(df.subj.nunique()),
        'trials': int(len(df)),
        'original_rt_mean_ms': float(df.rt_original_ms.mean()),
        'fixation_sum_mean_ms': float(df.rt_fixations_ms.mean()),
        'gap_mean_ms': float(df.gap_ms.mean()),
        'gap_median_ms': float(df.gap_ms.median()),
        'gap_p95_ms': float(df.gap_ms.quantile(.95)),
        'gap_negative': int((df.gap_ms < 0).sum()),
        'gap_positive': int((df.gap_ms > 0).sum()),
        'logrt_delta_mean': float(df.logrt_delta.mean()),
        'excluded_using_fixation_rt': int(df.excluded_using_fixation_rt.sum()),
        'excluded_using_recorded_rt': int(df.excluded_using_recorded_rt.sum()),
        'exclusion_disagreement': int((df.excluded_using_recorded_rt != df.excluded_using_fixation_rt).sum()),
        'within_subject_logrt_slopes': slopes,
        'note': 'Descriptive regression only, not inference. The current simulator consumes a fixation sequence and '
                'simulates RT independently; missing fixation-to-fixation time cannot be localized from this file. '
                'Neither RT definition was imputed or modified.',
    }

    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / 'trial_rt_comparison.csv', index=False)
    per_subject.to_csv(args.output / 'subject_rt_comparison.csv', index=False)
    with (args.output / 'rt_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Subjects={report['subjects']}; trials={report['trials']}")
    print(f"RT original/fixation sum mean: {report['original_rt_mean_ms']:.1f} / "
          f"{report['fixation_sum_mean_ms']:.1f} ms")
    print(f"RT gap mean/median: {report['gap_mean_ms']:.1f} / {report['gap_median_ms']:.1f} ms")
    print('maxT excluded by fixation/recorded RT:', report['excluded_using_fixation_rt'],
          '/', report['excluded_using_recorded_rt'])
    for key, value in slopes.items():
        print('Within-subject logRT slopes', key, value)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
