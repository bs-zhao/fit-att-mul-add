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
