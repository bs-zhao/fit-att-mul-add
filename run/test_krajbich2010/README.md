# Krajbich 2010 — adaptation checks (Thomas et al., 2019, Experiment 1)

This is a **testing-only** workspace. Do not modify `run/fit_nn/food_equal/` during validation. Once checks pass, create `run/fit_nn/krajbich2010/` by copying the existing pipeline and adapting file paths.

## Download the two source files

From https://github.com/glamlab/gaze-bias-differences/tree/master/data/krajbich_2010_natneuro/original:

- `data_nature2010.dta`
- `data_nature2010_codebook.md`

Place them here (local, Git-ignored):

```text
data/krajbich2010/original/data_nature2010.dta
data/krajbich2010/original/data_nature2010_codebook.md
```

## First step

From repository root:

```bash
python run/test_krajbich2010/s0_prepare_data.py
```

Only produces outputs in `outputs/krajbich2010/s0_prepare_data/`:

- `trial_eye.csv` — trial-level data with the same key fields expected by the existing binary food-choice loader
- `trial_diagnostics.csv` — each trial's raw recorded RT, sum of fixation durations, and their difference
- `subject_summary.csv` — subject-level QA
- `quality_report.json` — aggregate counts, value range, cutoff counts and encoding conventions

**Important:** The installed `c_bifood` simulator uses **choice 1=left, 0=right**, **fixation 1=left, 0=right**, and **`vs=[right_value,left_value]`**. Accordingly, the converter keeps source choice as-is, maps ROI `1→1` (left) and `2→0` (right), and sets `v0=rightrating`, `v1=leftrating`. Ratings are not rescaled. Source RT and fixation durations are in milliseconds.

The current `food_equal/tools/funcs.py` loader sets RT equal to the **sum of fixation durations**, which may differ from the actual experimental RT. This stage makes no assumptions about missing time; evaluate `trial_diagnostics.csv` before moving on. No SBI models are fit by this script.

## Step 1 — Compare RT definitions (tested on original data)

```bash
python run/test_krajbich2010/s1_rt_sensitivity.py
```

Files under `outputs/krajbich2010/s1_rt_sensitivity/`:

- `rt_report.json`: mean gaps, per-RT-definition exclusion counts and value effects
- `trial_rt_comparison.csv`: side-by-side trial results
- `subject_rt_comparison.csv`: by-subject checks

This compares source `rt_original_ms` against `rt_fixations_ms`; neither is overwritten. On 3,791 trials, original RT mean was 2,206.4 ms, fixation-sum mean 1,851.1 ms, and the mean difference was 355.3 ms. The within-subject logRT magnitude-effect coefficients were -0.0131 and -0.0157, respectively.

## Step 2 — Prior-predictive smoke test (NO full training)

The source `c_bifood` package must already be installed. The script imports the **existing** `food_equal/tools/model_info.py` prior ranges and `tools/funcs.make_params`, uses exactly the existing eight model functions and inputs, and does not modify any existing fit files.

Run this after Step 0 and Step 1:

```bash
python run/test_krajbich2010/s2_prior_predictive.py --n-subjects 5 --n-trials 20
```

Files under `outputs/krajbich2010/s2_prior_predictive/`:

- `model_summary.csv`: hit rates, simulated RT median/P95, observed RT medians
- `simulated_trials.csv`: simulated choice, hit, and RT for each sampled trial
- `prior_predictive_report.json`: machine-readable summaries and limitations

**Warnings:** Prior predictive simulation is not model fitting or evidence of a best model. libc's random-number state inside the Cython simulation is not fully controlled by the NumPy seed. The simulator's `extend_last=500000` may mutate arrays, so the test passes copies. Source trial RT is separate from fixation time. Do not run full SBI until the RT protocol and prior-predictive fit are reviewed.

## Step 3 — Matched-trial prior coverage and parameter scaling

After prior predictive smoke test:

```bash
python run/test_krajbich2010/s3_prior_coverage.py --n-source-subjects 4 --n-draws 8 --n-trials 24
```

Produces `outputs/krajbich2010/s3_prior_coverage/coverage_summary.csv`, `parameter_draws.csv`, and `coverage_report.json`. Uses identical selected empirical trials across all eight model families, samples model-specific prior parameters, and explores diagnostic `d` multipliers 1, 0.5, 0.25 and boundary `a` multipliers 1, 2. Each baseline configuration is preserved. All simulator invocations use the existing c_bifood package; no equations are modified. Values outside original priors are exploratory and **must not be silently used for SBI training**.

**Evaluation**: compare hit rate, simulated RT median, 25%-window coverage of recorded versus fixation-summed RT, and conditional choice agreement. No model inference should be drawn from this test. RT discrepancies remain unresolved; do not commence full SBI until the evidence supports a consistent observation protocol.

## Step 3b — Targeted accumulator coverage extension

The original 36,864-simulation diagnostic yielded very short ACC RT under the original d/a priors. To avoid repeating DDM tests, s3 now accepts `--models` to select just accumulator models. Test smaller d and larger a without changing the fitted model equations or prior definitions:

```bash
python run/test_krajbich2010/s3_prior_coverage.py --models aRACE_1 aRACE_2 aRACE_t aRACE_g --n-source-subjects 4 --n-draws 12 --n-trials 24 --d-scales 0.25 0.125 0.0625 --a-scales 2 3 4 --output outputs/krajbich2010/s3_prior_coverage_acc_extended
```

Review `coverage_summary.csv` and `parameter_draws.csv` for RT match, hit rate and choice consistency. These diagnostic multipliers do not change the official SBI priors. Very slow/no-hit trials are censored by `max_rt=14.5`, so never use hit-only RT alone as sufficient evidence of coverage.

## Step 3c — Validate a *shared* candidate prior against more subjects

The extended 4-subject diagnostic suggests one **common ACC** pair of scales,
`d×0.25, a×3`, rather than selecting a different scale for each gaze model.
This is a candidate for validation only, not a change to `model_info.py`.
For DDM, the previous diagnostic suggested a common `d×0.5, a×2` setting.

Run both commands from the project root:

```bash
python run/test_krajbich2010/s3_prior_coverage.py --models aDDM_1 aDDM_2 aDDM_t aDDM_g --n-source-subjects 12 --n-draws 16 --n-trials 40 --d-scales 0.5 --a-scales 2 --output outputs/krajbich2010/s3_prior_coverage_ddm_validation
python run/test_krajbich2010/s3_prior_coverage.py --models aRACE_1 aRACE_2 aRACE_t aRACE_g --n-source-subjects 12 --n-draws 16 --n-trials 40 --d-scales 0.25 --a-scales 3 --output outputs/krajbich2010/s3_prior_coverage_acc_validation
```

Each run has `4 × 12 × 16 × 40 = 30,720` simulation calls (61,440 total).
Both invocations use the same NumPy seed and the same selected subjects/trials.
Check both `coverage_summary.csv` and `parameter_draws.csv`:
mean and quantiles of hit rates across draws, RT ratio against **recorded RT**
and against **fixation-summed RT**, choice agreement, and coverage across subjects.

**Caution:** Even if RT coverage passes, the experimental RT is systematically
longer than the sum of option-fixation durations; fitting to original RT with
the current `ndt=0` model could attribute unobserved inter-fixation time to
decision accumulation. Keep that observation-model limitation explicit.

## Step 4 — Wide shared prior calibration (current preferred next step)

All four DDM variants **share the exact same d and a prior rectangle**; all
four ACC variants also share the exact same d and a rectangle. DDM and ACC
can differ from each other due to their different architectures. The test
compares original, common S3-scaled and two broader candidate rectangles.
It does not modify `tools/model_info.py` or any other production file.

Simulation cutoff is **ceil(max original RT in ms × (1 + margin))**, with
a default 20% margin. The current raw Krajbich 2010 maximum is **23,572 ms**,
giving **28,287 ms**, or 28.287 s, at a 20% margin. This replaces the
hard-coded 14.5 s cutoff **for this diagnostic only** and is also the
candidate `MAX_LEN_RT` for a future Krajbich-specific training pipeline.
The original food_equal 15 s setting remains unchanged.

Initial test:

```bash
python run/test_krajbich2010/s4_calibrate_shared_priors.py --n-subjects 4 --n-draws 12 --n-trials 12 --rt-margin 0.20
```

All outputs remain in
`outputs/krajbich2010/s4_calibrate_shared_priors/`, including
`candidate_ranking.csv`, `model_candidate_summary.csv`,
`draw_diagnostics.csv`, and `calibration_report.json`.

The screening requires, *for every gaze branch within an architecture*:
(1) no more than 10% trials faster than 300 ms, (2) no more than 10%
simulation timeouts, and (3) at least 5% of parameter draws with a
hit rate ≥ 0.90 and a simulated/recorded median RT ratio between 0.5
and 2.0. These are **initial diagnostic thresholds, not inferential criteria**.
Among candidates passing the thresholds, the script sorts by d×a rectangle
area, preferring broader coverage. Because candidates are finite, it does
not claim to find globally optimal limits. Inspect candidate-level results
before picking bounds, and increase sampling for validation.

**Important limitations:** A very wide rectangular uniform prior can put
substantial probability mass on unrealistic combinations even when the
marginal parameter ranges are individually plausible. A maximum RT is
sensitive to outliers. The diagnostic records both hit and timeout rates;
hit-only RT medians should never be used on their own. The `ndt=0`
observation protocol still differs from the experimental RT definition
and must be resolved before model fitting.

## Step 5 — Validate S4 candidate priors on more subjects (next run)

S4's 4-subject experiment found that `wide_2` passed its initial
**10% fast RT / 10% timeout** thresholds for all eight models; however,
the worst DDM timeout rate was 9.03% and the worst ACC fast-RT rate
was 6.77%. Do **not** finalize bounds using only four subjects.

S5 reruns the S4 simulator on the same selected source trials within
each candidate and model, comparing the three surviving candidates
`s3_shared`, `wide_1`, `wide_2` using stricter screening and
per-subject diagnostics.

From repository root:

```bash
git pull --ff-only
python run/test_krajbich2010/s5_validate_shared_priors.py --n-subjects 12 --n-draws 12 --n-trials 20
```

This makes `12 × 12 × 20 × 8 × 3 = 69,120` simulator calls. If the
validation remains viable, run an all-subject confirmation separately
using `--n-subjects 39` and a different `--output` directory.

Outputs under `outputs/krajbich2010/s5_validate_shared_priors/`:

- `candidate_ranking.csv`: worst branch on each screening criterion
- `model_summary.csv`: fast RT, timeouts, pathological draws, and RT coverage by model
- `subject_summary.csv`: differences between source subjects
- `validation_report.json`: configuration and simulation cutoff
- `s4_raw/`: unaggregated S4 results, including per-parameter-draw diagnostics

The preliminary S5 thresholds are 5% fast RT (<300 ms), 10% overall
timeouts, 15% pathological parameter draws (at least 25% timeouts
within a draw), and 20% parameter draws with a simulated/recorded
median RT ratio in [0.5,2] and hit rate >=90%. These are
diagnostic screening thresholds, not scientific or publication criteria.
The four gaze branches within each architecture share identical d/a
bounds. The script never edits the production priors. Recorded RT and
summed fixation-duration RT still require a consistent modeling decision.

## Step 5b — Targeted rectangles following the 12-subject S5 results

The 12-subject S5 validation found only `s3_shared` passed all the provisional
thresholds for both DDM and ACC; both `wide_1` and `wide_2` failed.
The current common baseline d/a ranges (identical across all four gaze
branches per architecture) are:

- DDM: d [0.05, 2.5], a [2, 10]
- ACC: d [0.025, 2], a [1.5, 15]

The next targeted test explores three specific rectangles per architecture,
defined in `s5_candidate_rectangles.json`:

- `raised_floor` increases a_min to suppress extremely fast decisions.
- `trimmed_ceiling` increases d_min and lowers a_max to limit no-hits.
- `balanced_wide` makes more modest changes in all four bounds to test a
  wider rectangle without the extreme tails of `wide_2`.

Run:

```bash
git pull --ff-only
python run/test_krajbich2010/s5_validate_shared_priors.py --candidate-config run/test_krajbich2010/s5_candidate_rectangles.json --candidate s3_shared raised_floor trimmed_ceiling balanced_wide --n-subjects 12 --n-draws 12 --n-trials 20 --output outputs/krajbich2010/s5_targeted_rectangles
```

This runs 92,160 simulator calls (8 models × 4 candidate rectangles × 12
subjects × 12 parameter draws × 20 trials). Inspect `candidate_ranking.csv`,
`model_summary.csv`, `subject_summary.csv` and `s4_raw/draw_diagnostics.csv`.
This is diagnostic only and does not change production priors, equations,
or the unresolved fixation-versus-recorded RT observation protocol.

## Step 5c — Holdout verification (27 subjects never selected in S5)

The 12-subject S5 targeted screening used exactly the following IDs:
`47 38 17 56 34 23 13 32 10 55 22 35`.
The original trial_eye.csv contains 39 subjects, so the remaining
**27** subjects can be used as independent input-sequence verification.
The S4 and S5 scripts now accept `--exclude-subjects` to prevent leakage.

From the repository root:

```bash
python run/test_krajbich2010/s5_validate_shared_priors.py --candidate-config run/test_krajbich2010/s5_candidate_rectangles.json --candidate s3_shared trimmed_ceiling balanced_wide --exclude-subjects 47 38 17 56 34 23 13 32 10 55 22 35 --n-subjects 27 --n-draws 8 --n-trials 20 --output outputs/krajbich2010/s5_holdout_validation
```

This runs 103,680 simulator calls (8 models × 3 rectangles × 27 subjects
× 8 parameter draws × 20 trials), and compares the *already-selected*
three candidates without updating them. The original source RT
maximum remains the basis for the 20%-margin simulation cutoff.
Verify that the result's `validation_report.json` lists exactly the
remaining 27 subjects and `candidate_ranking.csv` does not include
any held-out subject from the previous selection.

If balanced_wide fails the independent checks, the narrower
trimmed_ceiling remains a candidate due to its lower timeout rate.
If balanced_wide passes, this supports its *practical* prior coverage,
not theoretical optimality. The current fixation-versus-recorded RT
observation mismatch still prevents full model fitting without
an explicit protocol.
