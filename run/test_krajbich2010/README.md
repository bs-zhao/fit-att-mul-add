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
