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

**Important:** Source choice `1=left, 0=right` becomes `response 0=left, 1=right`. Source fixation ROI `1=left, 2=right` becomes `0=left, 1=right`. Ratings are not rescaled. Source RT and fixation durations are both in milliseconds.

The current `food_equal/tools/funcs.py` loader sets RT equal to the **sum of fixation durations**, which may differ from the actual experimental RT. This stage makes no assumptions about missing time; evaluate `trial_diagnostics.csv` before moving on. No SBI models are fit by this script.
