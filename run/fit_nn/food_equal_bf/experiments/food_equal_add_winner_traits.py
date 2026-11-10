#!/usr/bin/env python3
"""Exploratory subject-level behavior of eight-model DDM-Add vs RACE-Add winners.

Run from run/fit_nn/food_equal_bf on OSC, or from any directory with --input
and --s6 pointing to the original trial_eye.csv and eight-model S6 outputs.
No BayesFlow/GPU/old checkpoints needed; all computations use pandas/scipy.

Default S6 folder: sweep_v1_shared_d0p1to5p0_a1p0to5p0_theta0to1_gamma0to5.
The matching checks protect against comparing scores and behavior from
inconsistent trial filtering or a different source CSV.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

HERE = Path(__file__).resolve()
# When installed at run/fit_nn/food_equal_bf/experiments/*.py
ROOT = HERE.parents[4] if HERE.parent.name == 'experiments' else Path.cwd()
DEFAULT_EXPERIMENT = ('sweep_v1_shared_d0p1to5p0_'
                      'a1p0to5p0_theta0to1_gamma0to5')
DEFAULT_S6 = (ROOT / 'outputs/food_equal_bf/prior_sensitivity' /
              DEFAULT_EXPERIMENT / 'classifier/formal/real_data')
DEFAULT_INPUT = ROOT / 'data/food_equal/trial_eye.csv'
DDM, RACE = 'aDDM_g', 'aRACE_g'
LABELS = {'aDDM_g': 'DDM-Add', 'aRACE_g': 'RACE-Add'}
ALL_MODELS = ['aDDM_1', 'aDDM_2', 'aDDM_t', 'aDDM_g',
              'aRACE_1', 'aRACE_2', 'aRACE_t', 'aRACE_g']
REQUIRED = ['subj', 'v0', 'v1', 'response', 'arr_ml3_left', 'arr_ml3_time']

# These are descriptive between-subject endpoints, not independent trial data.
FEATURES = [
    'n_trials', 'mean_abs_value_diff', 'value_tie_rate',
    'rt_mean_s', 'rt_median_s', 'rt_sd_s', 'rt_p90_s', 'rt_cv',
    'rt_over_3s_rate', 'rt_vs_abs_value_diff_spearman',
    'nfix_mean', 'nfix_median', 'nfix_sd', 'switches_mean',
    'single_fixation_rate', 'mean_fixation_duration_ms',
    'gaze0_time_fraction', 'gaze0_first_rate', 'gaze0_last_rate',
    'off_target_time_fraction',
    'choice0_rate', 'choose_high_value_rate',
    'choose_high_value_easy_rate', 'choose_high_value_hard_rate',
    'choose_more_gazed_rate', 'chosen_gaze_time_fraction',
    'first_fixation_choice_match_rate', 'last_fixation_choice_match_rate',
    'value_gaze_conflict_rate', 'choose_high_value_on_conflict_rate',
    'choice0_on_value_ties',
    'choice_value_beta', 'choice_gaze_beta', 'choice_value_gaze_interaction_beta',
    'choice_intercept', 'rt_vs_nfix_spearman',
]
PLOT_FEATURES = [
    'rt_median_s', 'nfix_mean', 'choose_high_value_rate',
    'choose_more_gazed_rate', 'choice_value_beta', 'choice_gaze_beta',
]


def sid_key(value):
    """Canonical join key without assuming numeric subject identifiers."""
    s = str(value).strip()
    if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
        return s[:-2]
    return s


def read_s6(path):
    """Support an extracted S6 real_data directory, formal dir, or formal.zip."""
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            def find(name, required=True):
                matches = [x for x in names if x == name or x.endswith('/' + name)]
                if len(matches) > 1:
                    matches = [x for x in matches if '/real_data/' in '/' + x]
                if not matches:
                    if required:
                        raise FileNotFoundError(f'{name} missing in {path}')
                    return None
                if len(matches) > 1:
                    raise ValueError(f'Multiple ambiguous {name} in archive')
                return z.read(matches[0])
            probs = pd.read_csv(io.BytesIO(find('model_probabilities.csv')))
            audit = pd.read_csv(io.BytesIO(find('trial_audit.csv')))
            pair = pd.read_csv(io.BytesIO(find('pairwise_log_bayes_factors.csv')))
            metadata_bytes = find('metadata.json', required=False)
            metadata = json.loads(metadata_bytes) if metadata_bytes else None
    else:
        folder = path / 'real_data' if (path / 'real_data').is_dir() else path
        probs = pd.read_csv(folder / 'model_probabilities.csv')
        audit = pd.read_csv(folder / 'trial_audit.csv')
        pair = pd.read_csv(folder / 'pairwise_log_bayes_factors.csv')
        candidates = [folder.parent / 'metadata.json', folder / 'metadata.json']
        metadata_path = next((p for p in candidates if p.is_file()), None)
        metadata = json.loads(metadata_path.read_text()) if metadata_path else None
    for col in ['subj', 'best_model', *ALL_MODELS]:
        if col not in probs:
            raise ValueError(f'Expected eight-model classifier: missing {col}')
    if probs.subj.isna().any() or probs.subj.map(sid_key).duplicated().any():
        raise ValueError('Duplicate/missing S6 subject IDs')
    if not np.allclose(probs[ALL_MODELS].sum(axis=1).to_numpy(), 1, atol=1e-4):
        raise ValueError('Model probabilities do not sum to one')
    if not np.array_equal(probs[ALL_MODELS].idxmax(axis=1).to_numpy(),
                          probs.best_model.to_numpy()):
        raise ValueError('best_model inconsistent with eight model probabilities')
    bf = pair[(pair.numerator == RACE) & (pair.denominator == DDM)]
    if len(bf) != len(probs) or bf.subj.map(sid_key).duplicated().any():
        raise ValueError('S6 RACE-Add / DDM-Add pairwise log BF missing/duplicated')
    probs = probs.copy()
    probs['sid_key'] = probs.subj.map(sid_key)
    bf = bf[['subj', 'log_bayes_factor']].copy()
    bf['sid_key'] = bf.subj.map(sid_key)
    bf = bf.drop(columns='subj').rename(columns={
        'log_bayes_factor': 'logBF_RACEAdd_vs_DDMAdd'})
    probs = probs.merge(bf, on='sid_key', how='left', validate='one_to_one')
    if probs.logBF_RACEAdd_vs_DDMAdd.isna().any():
        raise ValueError('Missing pairwise BF after subject join')
    probs['p_RACEAdd_given_two_add'] = expit(probs.logBF_RACEAdd_vs_DDMAdd)
    probs['winner_group'] = np.select([probs.best_model.eq(DDM),
                                       probs.best_model.eq(RACE)],
                                      ['DDM-Add', 'RACE-Add'], default='Other')
    return probs, audit, metadata


def parse_int_array(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.array([], dtype=np.int64)
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=np.int64)
    s = str(x).strip()
    try:
        return np.asarray(ast.literal_eval(s), dtype=np.int64).reshape(-1)
    except (SyntaxError, ValueError, TypeError):
        s = s.strip('[]')
        if not s:
            return np.array([], dtype=np.int64)
        return np.fromstring(s.replace(',', ' '), sep=' ', dtype=np.int64)


def gaze_summary(pos, dur):
    """Replicate S6's treatment of gaze and merge consecutive identical codes."""
    mask = (dur > 0) & np.isin(pos, [0, 1])
    p, d = pos[mask], dur[mask]
    if len(p) == 0:
        return dict(nfix=0, switches=0, first=np.nan, last=np.nan,
                    time0=0., time1=0., eye_ms=0.)
    starts = np.r_[True, p[1:] != p[:-1]]
    run_count = int(starts.sum())
    t0 = float(d[p == 0].sum())
    t1 = float(d[p == 1].sum())
    return dict(nfix=run_count, switches=run_count-1,
                first=int(p[0]), last=int(p[-1]),
                time0=t0, time1=t1, eye_ms=t0+t1)


def load_trials(path, model_probs, expected_audit, metadata):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Original trial data not found: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata is not None and metadata.get('input_sha256') != digest:
        raise ValueError('Raw trial_eye.csv SHA256 does not match S4 checkpoint metadata')
    original = pd.read_csv(path)
    missing = [x for x in REQUIRED if x not in original.columns]
    if missing:
        raise ValueError(f'Missing raw data columns: {missing}')
    original['arr_pos'] = original.arr_ml3_left.map(parse_int_array)
    original['arr_du'] = original.arr_ml3_time.map(parse_int_array)
    # Same initial filter as food_equal/tools/funcs.py:load_food_data.
    valid_shape = [len(p) > 0 and len(p) == len(d)
                   for p, d in zip(original.arr_pos, original.arr_du)]
    df = original.loc[valid_shape].copy()
    df['sid_key'] = df.subj.map(sid_key)
    wanted = set(model_probs.sid_key)
    if not set(df.sid_key).issuperset(wanted):
        raise ValueError('S6 subject IDs not all found in original CSV')
    df = df[df.sid_key.isin(wanted)]
    retained, audit = [], []
    for sid, g in df.groupby('sid_key', sort=False):
        n_good = 0
        counts = dict(invalid_choice=0, invalid_rt=0, invalid_fixation=0,
                      invalid_values=0, maxT_excluded=0)
        for index, row in g.iterrows():
            pos, dur = row.arr_pos, row.arr_du
            rt = dur.sum() * .001
            try:
                choice = int(row.response)  # Same integer coding as load_food_data.
            except (TypeError, ValueError):
                choice = -99
            v0, v1 = float(row.v0), float(row.v1)
            reason = None
            if choice not in (0, 1):
                reason = 'invalid_choice'
            elif not np.isfinite(rt) or rt <= 0:
                reason = 'invalid_rt'
            elif len(pos) == 0 or len(pos) != len(dur) or np.any(dur < 0):
                reason = 'invalid_fixation'
            elif not np.isfinite([v0, v1]).all():
                reason = 'invalid_values'
            elif len(pos) >= 128 or np.ceil(dur.sum() / 3) >= np.floor(15000 / 3):
                reason = 'maxT_excluded'
            if reason is not None:
                counts[reason] += 1
                continue
            gaze = gaze_summary(pos, dur)
            n_eye = gaze['eye_ms']
            diff_v = v0-v1
            diff_gaze = (gaze['time0']-gaze['time1']) / n_eye if n_eye else np.nan
            selected_high = float(choice == (0 if diff_v > 0 else 1)) if diff_v != 0 else np.nan
            longer = (0 if diff_gaze > 0 else 1) if np.isfinite(diff_gaze) and diff_gaze != 0 else -1
            conflict = (diff_v * diff_gaze < 0) if np.isfinite(diff_gaze) and diff_v != 0 else False
            retained.append({
                'subj': row.subj, 'sid_key': sid, 'row_index': index,
                'v0': v0, 'v1': v1, 'value_difference0minus1': diff_v,
                'abs_value_gap': abs(diff_v), 'choice': choice, 'rt_s': rt,
                'nfix': gaze['nfix'], 'switches': gaze['switches'],
                'first_gaze': gaze['first'], 'last_gaze': gaze['last'],
                'gaze0_ms': gaze['time0'], 'gaze1_ms': gaze['time1'],
                'eye_ms': n_eye, 'total_ms': float(dur.sum()),
                'gaze_difference0minus1': diff_gaze,
                'gaze0_fraction': gaze['time0']/n_eye if n_eye else np.nan,
                'off_target_fraction': max(0, dur.sum()-n_eye)/dur.sum(),
                'first_matches_choice': float(gaze['first'] == choice) if n_eye else np.nan,
                'last_matches_choice': float(gaze['last'] == choice) if n_eye else np.nan,
                'chosen_gaze_fraction': (gaze['time0'] if choice == 0 else gaze['time1'])/n_eye if n_eye else np.nan,
                'choose_high_value': selected_high,
                'choose_more_gazed': float(choice == longer) if longer != -1 else np.nan,
                'value_gaze_conflict': float(conflict) if np.isfinite(diff_gaze) and diff_v != 0 else np.nan,
                'choose_high_value_on_conflict': selected_high if conflict else np.nan,
                'fixation_duration_mean_ms': n_eye/gaze['nfix'] if gaze['nfix'] else np.nan,
            })
            n_good += 1
        audit.append(dict(sid_key=sid, input_trials=len(g), retained_trials=n_good,
                          excluded_trials=len(g)-n_good, **counts))
    out = pd.DataFrame(retained)
    recomputed_audit = pd.DataFrame(audit).set_index('sid_key')
    observed_audit = expected_audit.copy()
    observed_audit['sid_key'] = observed_audit.subj.map(sid_key)
    observed_audit = observed_audit.set_index('sid_key')
    for sid in wanted:
        if sid not in recomputed_audit.index or sid not in observed_audit.index:
            raise ValueError(f'Audit missing subject {sid}')
        for col in ['input_trials', 'retained_trials', 'excluded_trials',
                    'invalid_choice', 'invalid_rt', 'invalid_fixation',
                    'invalid_values', 'maxT_excluded']:
            ours, theirs = int(recomputed_audit.loc[sid, col]), int(observed_audit.loc[sid, col])
            if ours != theirs:
                raise ValueError(f'Audit mismatch for subject {sid}, {col}: '
                                 f'behavior={ours}, S6={theirs}; do not compare mismatched data')
    if len(out) != int(expected_audit.retained_trials.sum()):
        raise ValueError('Total retained trial count differs from S6')
    return out, digest


def rate(s):
    values = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def safe_sd(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1)) if len(a) > 1 else np.nan


def safe_spearman(x, y):
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or len(np.unique(a[m])) < 2 or len(np.unique(b[m])) < 2:
        return np.nan
    return float(spearmanr(a[m], b[m]).statistic)


def fit_choice_style(trials, scales):
    """Ridge-logistic choice0 ~ value difference + gaze difference + interaction.

    Predictors are standardized using GLOBAL SDs over retained trials. This
    yields comparable subject coefficients. This is descriptive, not causal.
    Penalty 0.5 * 0.5 * sum(beta_slopes^2) avoids complete separation.
    """
    x = trials[['value_difference0minus1', 'gaze_difference0minus1']].to_numpy(dtype=float, copy=True)
    x[:, 1] = np.nan_to_num(x[:, 1], nan=0.0)
    interaction = x[:, 0] * x[:, 1]
    raw = np.column_stack((x, interaction))
    for j in range(3):
        raw[:, j] = raw[:, j] / scales[j]
    raw = np.column_stack([np.ones(len(trials)), raw])
    y = (trials.choice.to_numpy(dtype=int) == 0).astype(float)
    def fun(beta):
        z = raw @ beta
        pred = expit(z)
        value = (np.logaddexp(0, z) - y*z).sum() + .25 * (beta[1:]**2).sum()
        grad = raw.T @ (pred-y)
        grad[1:] += .5*beta[1:]
        return value, grad
    init = np.array([np.log(np.clip(y.mean(), 1e-4, 1-1e-4) /
                             (1-np.clip(y.mean(), 1e-4, 1-1e-4))), 0., 0., 0.])
    res = minimize(fun, init, method='L-BFGS-B', jac=True)
    if not res.success or not np.all(np.isfinite(res.x)):
        warnings.warn(f'Logistic fit failed: {res.message}', stacklevel=2)
        return [np.nan]*4
    return [float(z) for z in res.x]


def subject_metrics(trials, probs):
    gabs = trials.loc[trials.abs_value_gap > 0, 'abs_value_gap']
    median_gap = float(np.median(gabs)) if len(gabs) else 0.
    x0 = trials.value_difference0minus1.to_numpy(dtype=float)
    x1 = np.nan_to_num(trials.gaze_difference0minus1.to_numpy(dtype=float), nan=0.)
    scales = np.std(np.column_stack([x0, x1, x0*x1]), axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.
    rows = []
    for sid, g in trials.groupby('sid_key', sort=False):
        vals = g.rt_s.to_numpy(dtype=float)
        non_tie = g[g.abs_value_gap > 0]
        high_gap = non_tie[non_tie.abs_value_gap >= median_gap]
        low_gap = non_tie[non_tie.abs_value_gap < median_gap]
        beta0, bval, bgaze, bint = fit_choice_style(g, scales)
        rows.append(dict(
            sid_key=sid, n_trials=len(g),
            mean_abs_value_diff=float(g.abs_value_gap.mean()),
            value_tie_rate=float((g.abs_value_gap == 0).mean()),
            rt_mean_s=float(np.mean(vals)), rt_median_s=float(np.median(vals)),
            rt_sd_s=safe_sd(vals), rt_p90_s=float(np.quantile(vals, .9)),
            rt_cv=safe_sd(vals) / np.mean(vals) if np.mean(vals) > 0 else np.nan,
            rt_over_3s_rate=float(np.mean(vals > 3)),
            rt_vs_abs_value_diff_spearman=safe_spearman(g.rt_s, g.abs_value_gap),
            nfix_mean=float(g.nfix.mean()), nfix_median=float(g.nfix.median()),
            nfix_sd=safe_sd(g.nfix), switches_mean=float(g.switches.mean()),
            single_fixation_rate=float((g.nfix == 1).mean()),
            mean_fixation_duration_ms=rate(g.fixation_duration_mean_ms),
            gaze0_time_fraction=rate(g.gaze0_fraction),
            gaze0_first_rate=rate(g.first_gaze.eq(0).where(g.first_gaze.notna(), np.nan)),
            gaze0_last_rate=rate(g.last_gaze.eq(0).where(g.last_gaze.notna(), np.nan)),
            off_target_time_fraction=float(g.off_target_fraction.mean()),
            choice0_rate=float((g.choice == 0).mean()),
            choose_high_value_rate=rate(non_tie.choose_high_value),
            choose_high_value_easy_rate=rate(high_gap.choose_high_value),
            choose_high_value_hard_rate=rate(low_gap.choose_high_value),
            choose_more_gazed_rate=rate(g.choose_more_gazed),
            chosen_gaze_time_fraction=rate(g.chosen_gaze_fraction),
            first_fixation_choice_match_rate=rate(g.first_matches_choice),
            last_fixation_choice_match_rate=rate(g.last_matches_choice),
            value_gaze_conflict_rate=rate(g.value_gaze_conflict),
            choose_high_value_on_conflict_rate=rate(g.choose_high_value_on_conflict),
            choice0_on_value_ties=rate(g.loc[g.abs_value_gap == 0, 'choice'].eq(0)),
            choice_value_beta=bval, choice_gaze_beta=bgaze,
            choice_value_gaze_interaction_beta=bint, choice_intercept=beta0,
            rt_vs_nfix_spearman=safe_spearman(g.rt_s, g.nfix),
        ))
    df = pd.DataFrame(rows)
    return probs.merge(df, on='sid_key', how='left', validate='one_to_one')


def hedges_g(ddm, race):
    n0, n1 = len(ddm), len(race)
    if n0 < 2 or n1 < 2:
        return np.nan
    pooled = ((n0-1)*ddm.var(ddof=1)+(n1-1)*race.var(ddof=1))/(n0+n1-2)
    if pooled <= 0:
        return np.nan
    raw = (race.mean()-ddm.mean()) / np.sqrt(pooled)
    return float(raw * (1-3/(4*(n0+n1)-9)))


def cliffs_delta(ddm, race):
    return float(np.sign(race[:, None] - ddm[None, :]).mean())


def bh_adjust(values):
    x = np.asarray(values, float)
    out = np.full(len(x), np.nan)
    valid = np.isfinite(x)
    if not valid.any():
        return out
    order = np.argsort(x[valid])
    sorted_p = x[valid][order]
    adjusted = np.minimum.accumulate((sorted_p*len(sorted_p)/
                                      np.arange(1, len(sorted_p)+1))[::-1])[::-1]
    temp = np.empty(len(order))
    temp[order] = np.clip(adjusted, 0, 1)
    out[valid] = temp
    return out


def group_comparisons(subjects, n_boot, n_permute, seed):
    ddm = subjects[subjects.winner_group == 'DDM-Add']
    race = subjects[subjects.winner_group == 'RACE-Add']
    if len(ddm) < 2 or len(race) < 2:
        raise ValueError('At least 2 winners per Add model required')
    rng = np.random.default_rng(seed)
    tests, summary = [], []
    for col in FEATURES:
        a = ddm[col].to_numpy(dtype=float); a = a[np.isfinite(a)]
        b = race[col].to_numpy(dtype=float); b = b[np.isfinite(b)]
        for group, values in [('DDM-Add', a), ('RACE-Add', b)]:
            summary.append(dict(feature=col, group=group, n=len(values),
                                mean=float(values.mean()) if len(values) else np.nan,
                                sd=safe_sd(values),
                                median=float(np.median(values)) if len(values) else np.nan))
        stat = dict(feature=col, n_ddm=len(a), n_race=len(b),
                    mean_ddm=float(a.mean()) if len(a) else np.nan,
                    mean_race=float(b.mean()) if len(b) else np.nan,
                    race_minus_ddm=float(b.mean()-a.mean()) if len(a) and len(b) else np.nan,
                    hedges_g=np.nan, cliffs_delta=np.nan,
                    welch_p=np.nan, mann_whitney_p=np.nan,
                    permutation_p=np.nan, bootstrap_ci_low=np.nan,
                    bootstrap_ci_high=np.nan)
        if len(a) >= 3 and len(b) >= 3:
            stat['hedges_g'] = hedges_g(a,b)
            stat['cliffs_delta'] = cliffs_delta(a,b)
            stat['welch_p'] = float(ttest_ind(a,b,equal_var=False).pvalue)
            stat['mann_whitney_p'] = float(mannwhitneyu(a,b,alternative='two-sided').pvalue)
            pooled = np.r_[a,b]
            observed = abs(b.mean()-a.mean())
            extreme = 0
            for _ in range(n_permute):
                perm = rng.permutation(len(pooled))
                contrast = abs(pooled[perm[len(a):]].mean()-pooled[perm[:len(a)]].mean())
                extreme += contrast >= observed - 1e-14
            stat['permutation_p'] = (extreme+1)/(n_permute+1)
            d = rng.choice(a, (n_boot,len(a)), replace=True).mean(axis=1)
            r = rng.choice(b, (n_boot,len(b)), replace=True).mean(axis=1)
            stat['bootstrap_ci_low'], stat['bootstrap_ci_high'] = [float(q) for q in np.quantile(r-d,[.025,.975])]
        tests.append(stat)
    df = pd.DataFrame(tests)
    df['permutation_q_BH'] = bh_adjust(df.permutation_p)
    df['welch_q_BH'] = bh_adjust(df.welch_p)
    return df, pd.DataFrame(summary)


def continuous_correlations(subjects):
    result = []
    for col in FEATURES:
        x = subjects[col].to_numpy(dtype=float)
        y = subjects.logBF_RACEAdd_vs_DDMAdd.to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 7 and np.unique(x[valid]).size > 1:
            s = spearmanr(x[valid],y[valid])
            rho, p = float(s.statistic), float(s.pvalue)
        else:
            rho, p = np.nan, np.nan
        result.append(dict(feature=col,n=int(valid.sum()),spearman_rho=rho,p=p))
    df=pd.DataFrame(result)
    df['q_BH']=bh_adjust(df.p)
    return df


def plot_6(subjects, out, against='winner_group'):
    rng = np.random.default_rng(812)
    colors = ['C0','C1']
    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    for ax, col in zip(axes.flat, PLOT_FEATURES):
        groups = [subjects.loc[subjects.winner_group==g,col].dropna().to_numpy(dtype=float)
                  for g in ('DDM-Add','RACE-Add')]
        ax.boxplot(groups, tick_labels=['DDM-Add','RACE-Add'], showfliers=False,
                   medianprops={'color':'black'})
        for j, (c,values) in enumerate(zip(colors,groups),start=1):
            ax.scatter(j+rng.uniform(-.12,.12,len(values)), values,
                       s=18, alpha=.68, color=c)
        ax.set_title(col.replace('_',' '),fontsize=10)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Behavior by eight-model Top-1 winners (subjects are independent units)')
    fig.tight_layout()
    fig.savefig(out/'01_winner_group_behavior.png',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,col in zip(axes.flat,PLOT_FEATURES):
        for group,color in zip(['DDM-Add','RACE-Add','Other'],['C0','C1','gray']):
            p=subjects[subjects.winner_group==group]
            ax.scatter(p[col],p.logBF_RACEAdd_vs_DDMAdd,label=group,
                       s=32,alpha=.8,color=color)
        ax.axhline(0,linestyle='--',color='black',linewidth=.7)
        ax.set(xlabel=col.replace('_',' '),ylabel='log BF(RACE-Add / DDM-Add)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'02_behavior_vs_continuous_logBF.png',dpi=180)
    plt.close(fig)


def plot_trial_curves(trials,out):
    x = trials.loc[trials.abs_value_gap > 0].copy()
    if x.abs_value_gap.nunique() < 2:
        return
    bins = pd.qcut(x.abs_value_gap,4,duplicates='drop')
    x['difficulty_bin'] = bins.astype(str)
    labels = list(dict.fromkeys(x.difficulty_bin))
    labels.sort(key=lambda name: x.loc[x.difficulty_bin==name, 'abs_value_gap'].median())
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for group,color in [('DDM-Add','C0'),('RACE-Add','C1')]:
        subset=x[x.winner_group==group]
        for ax,metric in zip(axs,['choose_high_value','rt_s']):
            by_subj=subset.groupby(['sid_key','difficulty_bin'])[metric].mean().reset_index()
            agg=by_subj.groupby('difficulty_bin')[metric].agg(['mean','count','std'])
            vals=np.array([agg.loc[b,'mean'] if b in agg.index else np.nan for b in labels])
            se=np.array([agg.loc[b,'std']/np.sqrt(agg.loc[b,'count']) if b in agg.index and agg.loc[b,'count']>1 else np.nan for b in labels])
            ax.errorbar(range(len(labels)),vals,yerr=se,marker='o',label=group,color=color,capsize=3)
    for ax,title in zip(axs,['Choose higher-valued option','Mean RT (s)']):
        ax.set(xticks=range(len(labels)),xticklabels=labels,xlabel='|v0 - v1| bins',title=title)
        ax.grid(alpha=.2)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out/'03_value_difficulty_choice_and_rt.png',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT,help='Original trial_eye.csv')
    p.add_argument('--s6',type=Path,default=DEFAULT_S6,help='Eight-model S6 real_data directory, formal directory, or ZIP')
    p.add_argument('--output',type=Path,help='Separate output folder (default: sibling analysis/add_winner_traits)')
    p.add_argument('--bootstrap',type=int,default=3000)
    p.add_argument('--permutations',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261110)
    a=p.parse_args()
    if a.bootstrap < 100 or a.permutations < 100:
        p.error('--bootstrap and --permutations must both be >=100')
    s6,audit,meta=read_s6(a.s6)
    trials,digest=load_trials(a.input,s6,audit,meta)
    subjects=subject_metrics(trials,s6)
    if subjects.n_trials.isna().any():
        raise ValueError('Some S6 subjects have no retained behavioral trials')
    trials=trials.merge(s6[['sid_key','winner_group','logBF_RACEAdd_vs_DDMAdd']],
                        on='sid_key',how='left',validate='many_to_one')
    tests,summary=group_comparisons(subjects,a.bootstrap,a.permutations,a.seed)
    corrs=continuous_correlations(subjects)
    if a.output:
        out=a.output.expanduser().resolve()
    elif a.s6.is_dir():
        parent=a.s6.resolve()
        if parent.name=='real_data':parent=parent.parent
        out=parent/'analysis/add_winner_traits'
    else:
        out=Path.cwd()/'add_winner_traits'
    out.mkdir(parents=True,exist_ok=True)
    subjects.to_csv(out/'subject_features.csv',index=False)
    trials.to_csv(out/'trial_features.csv',index=False)
    summary.to_csv(out/'group_summary.csv',index=False)
    tests.to_csv(out/'group_comparisons.csv',index=False)
    corrs.to_csv(out/'all_subjects_logBF_correlations.csv',index=False)
    plot_6(subjects,out)
    plot_trial_curves(trials,out)
    group_counts=subjects.winner_group.value_counts().to_dict()
    report={
        'input_sha256':digest,
        's6_source':str(a.s6),
        'n_subjects':len(subjects),
        'n_trials':len(trials),
        'winner_group_counts':group_counts,
        'mean_logBF_RACEAdd_vs_DDMAdd':float(subjects.logBF_RACEAdd_vs_DDMAdd.mean()),
        'race_two_model_pairwise_wins':int((subjects.logBF_RACEAdd_vs_DDMAdd>0).sum()),
        'ddm_two_model_pairwise_wins':int((subjects.logBF_RACEAdd_vs_DDMAdd<0).sum()),
        'effect_direction':'RACE-Add minus DDM-Add for all group-comparison effects',
        'sampling_unit':'participant, not trial',
        'preprocessing':'Same initial shape, binary response, RT and maxT filters as food_equal_bf S6; validated against S6 trial_audit.csv',
        'choice_code':'choice=0 selects v0; choice=1 selects v1; gaze 0/1 refer to corresponding option indices',
        'logistic':'ridge logistic choice0 ~ standardized (v0-v1) + standardized gaze0-gaze1 duration fraction + standardized interaction; predictors globally scaled; L2 penalty 0.5',
        'inference':'Exploratory analyses; BH-FDR across features, no causal inference; no feature-based changes to S6 labels',
        'model_prior':meta.get('model_prior') if meta else None,
    }
    (out/'analysis_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=float))
    rank=tests.sort_values('permutation_p').head(12)
    lines=[
        'FOOD EQUAL — behavioral signatures of DDM-Add vs RACE-Add winners',
        f'S6 source: {a.s6}',
        f'Participants={len(subjects)}  retained trials={len(trials)}',
        f'Top-1 winner groups: {group_counts}',
        f'RACE-Add vs DDM-Add pairwise BF positive={(subjects.logBF_RACEAdd_vs_DDMAdd>0).sum()}',
        'Statistics compare PARTICIPANT LEVEL means. Effect direction: RACE minus DDM.',
        'No p-value is a model-selection criterion; groups are selected from the same behavioral data.',
        '',
        'Features ranked by exploratory subject-level permutation p-value:',
    ]
    for _,r in rank.iterrows():
        lines.append(f"{r.feature:43} DDM={r.mean_ddm:8.3f}  RACE={r.mean_race:8.3f} "
                     f"diff={r.race_minus_ddm:+8.3f}  g={r.hedges_g:+6.2f} "
                     f"perm_p={r.permutation_p:.4g} FDR_q={r.permutation_q_BH:.4g}")
    lines.extend(['','CAVEATS:','* Choice and gaze metrics are associations, not causal gaze effects.',
                  '* The model classification is derived from the same trials; observed group differences are exploratory.',
                  '* A global RT or gaze effect does not imply a model fitted it better.',
                  '* Non-significant p/q does not establish groups are identical.'])
    (out/'analysis_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:10]),flush=True)
    print('OUTPUT',out,flush=True)


if __name__=='__main__':
    main()