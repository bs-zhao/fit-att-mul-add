"""Validate and convert Krajbich et al. (2010) fixation-level Stata data.

Run from any directory:
    python run/test_krajbich2010/s0_prepare_data.py

Requires data/krajbich2010/original/data_nature2010.dta.
Outputs only to outputs/krajbich2010/s0_prepare_data/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'data/krajbich2010/original/data_nature2010.dta'
DEFAULT_OUTPUT = ROOT / 'outputs/krajbich2010/s0_prepare_data'
REQUIRED = ('subject', 'trial', 'fix_num', 'roi', 'event_duration',
            'choice', 'leftrating', 'rightrating', 'rt')


def to_json_number(value):
    return int(value) if isinstance(value, (int, np.integer)) else float(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f'Missing original Stata file: {args.input}\n'
            'Download data_nature2010.dta from '
            'https://github.com/glamlab/gaze-bias-differences/tree/master/'
            'data/krajbich_2010_natneuro/original'
        )

    raw = pd.read_stata(args.input, convert_categoricals=False)
    names = {str(col).lower(): col for col in raw.columns}
    missing = [col for col in REQUIRED if col not in names]
    if missing:
        raise ValueError(f'Missing Stata fields {missing}; available: {list(raw.columns)}')

    df = raw.rename(columns={names[col]: col for col in REQUIRED}).copy()
    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    invalid = df[list(REQUIRED)].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f'{int(invalid.sum())} raw fixation rows have missing/non-numeric required values; '
                         f'first row indices: {df.index[invalid].tolist()[:10]}')
    if not df['roi'].isin([1, 2]).all():
        values = df.loc[~df['roi'].isin([1, 2]), 'roi'].value_counts().to_dict()
        raise ValueError(f'Unexpected ROI codes {values}; expected 1=left, 2=right. '
                         'Inspect before deciding how to encode non-option gaze.')
    if not df['choice'].isin([0, 1]).all():
        raise ValueError(f'Unexpected choice codes: {sorted(df.choice.unique().tolist())}')
    if (df['event_duration'] <= 0).any() or (df['rt'] <= 0).any():
        raise ValueError('Nonpositive fixation duration or reaction time found.')
    if (df['fix_num'] % 1 != 0).any():
        raise ValueError('Noninteger fixation indices found.')

    prepared, diagnostics = [], []
    for (subject, trial), group in df.groupby(['subject', 'trial'], sort=False):
        group = group.sort_values('fix_num', kind='stable')
        if group['fix_num'].duplicated().any():
            raise ValueError(f'Duplicate fixation indices: subject={subject}, trial={trial}')
        for field in ('choice', 'leftrating', 'rightrating', 'rt'):
            if group[field].nunique() != 1:
                raise ValueError(f'{field} changes within subject={subject}, trial={trial}')

        # Stata codebook: choice=1 left, 0 right; roi=1 left, 2 right.
        # c_bifood simulator uses choice=1 left, 0 right; f=1 left, 0 right,
        # and values ordered [right, left]. Keep data aligned to the SIMULATOR.
        choice = int(group['choice'].iloc[0])
        positions = (2 - group['roi'].to_numpy(dtype=int)).tolist()
        durations = np.rint(group['event_duration'].to_numpy(dtype=float)).astype(int)
        if (durations <= 0).any():
            raise ValueError(f'Rounded fixation duration is nonpositive: subject={subject}, trial={trial}')

        original_rt_ms = float(group['rt'].iloc[0])
        fixation_rt_ms = int(durations.sum())
        prepared.append({
            'subj': to_json_number(subject),
            'trial': to_json_number(trial),
            'v0': float(group['rightrating'].iloc[0]),
            'v1': float(group['leftrating'].iloc[0]),
            'response': choice,
            'arr_ml3_left': str(positions),
            'arr_ml3_time': str(durations.tolist()),
            'rt_original_ms': original_rt_ms,
            'rt_fixations_ms': fixation_rt_ms,
        })
        diagnostics.append({
            'subj': to_json_number(subject),
            'trial': to_json_number(trial),
            'n_fixations': len(positions),
            'rt_original_ms': original_rt_ms,
            'rt_fixations_ms': fixation_rt_ms,
            'rt_gap_ms': original_rt_ms - fixation_rt_ms,
            'first_fix_num': int(group['fix_num'].iloc[0]),
            'last_fix_num': int(group['fix_num'].iloc[-1]),
        })

    prepared = pd.DataFrame(prepared)
    diag = pd.DataFrame(diagnostics)
    if prepared.empty:
        raise ValueError('No trials found in source data.')

    subj = diag.groupby('subj', sort=False).agg(
        n_trials=('trial', 'size'),
        mean_original_rt_ms=('rt_original_ms', 'mean'),
        mean_fixation_rt_ms=('rt_fixations_ms', 'mean'),
        mean_rt_gap_ms=('rt_gap_ms', 'mean'),
        mean_n_fixations=('n_fixations', 'mean'),
    ).reset_index()

    rt_gap = diag['rt_gap_ms'].to_numpy(dtype=float)
    report = {
        'source': 'Thomas et al. (2019), experiment 1: Krajbich et al. (2010)',
        'input_file': str(args.input),
        'raw_fixation_rows': int(len(df)),
        'subjects': int(prepared['subj'].nunique()),
        'trials': int(len(prepared)),
        'trials_per_subject_min': int(subj['n_trials'].min()),
        'trials_per_subject_max': int(subj['n_trials'].max()),
        'item_value_min': float(prepared[['v0', 'v1']].min().min()),
        'item_value_max': float(prepared[['v0', 'v1']].max().max()),
        'original_rt_max_ms': float(diag['rt_original_ms'].max()),
        'fixation_rt_max_ms': int(diag['rt_fixations_ms'].max()),
        'original_rt_at_least_15000_ms': int((diag['rt_original_ms'] >= 15000).sum()),
        'fixation_rt_at_least_15000_ms': int((diag['rt_fixations_ms'] >= 15000).sum()),
        'fixations_at_least_128': int((diag['n_fixations'] >= 128).sum()),
        'rt_gap_median_ms': float(np.median(rt_gap)),
        'rt_gap_p95_ms': float(np.percentile(rt_gap, 95)),
        'rt_gap_abs_gt_50_ms': int((np.abs(rt_gap) > 50).sum()),
        'rt_gap_negative_count': int((rt_gap < 0).sum()),
        'choice_encoding': '1=left, 0=right (same as source; matches c_bifood)',
        'fixation_encoding': '1=left, 0=right (2 - source ROI; matches c_bifood)',
        'value_encoding': 'v0=right, v1=left (matches c_bifood values=[right,left])',
        'time_unit': 'milliseconds for arr_ml3_time and rt_*_ms',
        'note': 'No missing inter-fixation or pre-fixation time was imputed. '
                'Existing load_food_data derives RT from sum of fixation durations, '
                'not the recorded rt_original_ms. Inspect trial_diagnostics.csv before model fitting.',
    }

    args.output.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output / 'trial_eye.csv', index=False)
    diag.to_csv(args.output / 'trial_diagnostics.csv', index=False)
    subj.to_csv(args.output / 'subject_summary.csv', index=False)
    with (args.output / 'quality_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Subjects: {report['subjects']}; trials: {report['trials']}; "
          f"source fixation rows: {report['raw_fixation_rows']}")
    print(f"Values: {report['item_value_min']} to {report['item_value_max']}")
    print(f"Original RT >= 15 s: {report['original_rt_at_least_15000_ms']}; "
          f"fixation sum >= 15 s: {report['fixation_rt_at_least_15000_ms']}")
    print(f"Median (original RT - fixation sum): {report['rt_gap_median_ms']:.1f} ms")
    print(f"Saved outputs to: {args.output}")
    if report['rt_gap_abs_gt_50_ms']:
        print('WARNING: Original RT and summed fixation durations differ by >50 ms '
              f"on {report['rt_gap_abs_gt_50_ms']} trials. Inspect diagnostics before fitting.")


if __name__ == '__main__':
    main()
