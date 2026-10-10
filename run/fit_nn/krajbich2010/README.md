# Krajbich 2010 isolated 8-model SBI pipeline

Mirrors `run/fit_nn/food_equal/` with unchanged model equations and file names.
Data and outputs never overlap food_equal. Execute commands below from repository
root; each entry script sets its own working directory.

## Input and RT protocol

Prepare `data/krajbich2010/trial_eye.csv` from
`data/krajbich2010/original/data_nature2010.dta` using
`python run/test_krajbich2010/s0_prepare_data.py`.

**Scheme A:** empirical RT for SBI equals the **sum of option fixation
durations**. The original recorded trial RT is preserved separately as
`rt_original_ms` in `trial_eye.csv` and the exported
`outputs/krajbich2010/s3_fe1_maxT/real_data/df.csv`. It is not
used in `fe.pkl` or trained decoders. The experimental RT includes extra
time not represented in fixation-event sequences; do not treat model results
as fits to full recorded RT. Simulated RT remains simulated decision time,
with fixed `ndt=0` from the original equations.

Encoding: `vs=[right,left]`, response 1=left/0=right,
and fixation position 1=left/0=right.

## Priors and limits

Eight models are enabled (DDM/ACC × NoAtt/AttOnly/Mul/Add).
All four branches within DDM share `d~U(0.06,2.80)`, `a~U(2.30,10.50)`.
All four ACC branches share `d~U(0.035,2.30)`, `a~U(2.00,14.50)`.
Theta/gamma prior ranges are unchanged from food_equal.
Maximum simulation time `28.287 s`; fixed sequence maxT `28287 ms`,
based on the largest recorded RT plus 20%; max number fixations 128.
These are screened prior candidates, not evidence of optimality.

## Prepare / smoke-test

```bash
python run/test_krajbich2010/s0_prepare_data.py
python run/fit_nn/krajbich2010/s5_fe_real_data_maxT.py
python run/fit_nn/krajbich2010/s1_gen1w.py --models aDDM_1 --n-round 2 --n-gen 1 --for-test
python run/fit_nn/krajbich2010/s5_fe_test_data_maxT.py
```

Smoke output is in `outputs/krajbich2010/s1_gen1_test`;
it does not fill the full training set.

## Main simulator → feature extraction → flow SBI

```bash
python run/fit_nn/krajbich2010/s1_gen1w.py
python run/fit_nn/krajbich2010/s2_getmaxRT.py
python run/fit_nn/krajbich2010/s3_fe1.py
python run/fit_nn/krajbich2010/s4_train_range_huber.py
```

Full simulation uses 8 models × 50 files × 500 synthetic subjects.
`s4_train_range_huber.py` now trains all eight models; it uses
**data2param_flow**, not the older classifier package.

## Posterior predictive model comparison

Only after all eight parameter decoders are trained:

```bash
python run/fit_nn/krajbich2010/compare_regen/s1_regen.py
python run/fit_nn/krajbich2010/compare_regen/s2_getmaxRT.py
python run/fit_nn/krajbich2010/compare_regen/s3_fe1_maxT.py
python run/fit_nn/krajbich2010/compare_regen/s4_train_classifier_maxT.py
python run/fit_nn/krajbich2010/compare_regen/s6_decode_maxT.py
```

Model classifier is the **legacy data2param** implementation and contains
all eight models. For MR, generate `s1_gen1w.py --for-test` and
`s5_fe_test_data_maxT.py`, train an MR classifier on independently
regenerated `--for-test` data, and decode with `s6_decode_maxT_mr.py`.
Detailed MR switches are documented inside the compare_regen scripts.

All production outputs: `outputs/krajbich2010/`. The prior-predictive and calibration test artifacts live exclusively under `outputs/krajbich2010/tests/`. No existing `food_equal`
training files or models are touched.

## Before any full run: preflight

From repository root:

```bash
git pull --ff-only
python run/fit_nn/krajbich2010/preflight.py
```

The preflight parses all scripts, verifies the shared priors and Scheme-A
RT, confirms the raw recorded RT column is preserved and reports any missing
`c_bifood`, `data2param_flow`, or `data2param` package. No outputs
are written. **A GitHub commit is not a completed runtime test**:
run this and the small smoke generation before full-scale simulation.

## Recommended OSC Slurm workflow

```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
python run/fit_nn/krajbich2010/preflight.py
sbatch run/fit_nn/krajbich2010/simulate_array.sbatch
```

Wait until all 8 Slurm array tasks have finished and each model has
`gen1.pkl` through `gen50.pkl`; do not start feature extraction prematurely.

```bash
python run/fit_nn/krajbich2010/s2_getmaxRT.py
sbatch run/fit_nn/krajbich2010/features_array.sbatch
```

Wait until all eight feature-extraction jobs finish; then:

```bash
sbatch run/fit_nn/krajbich2010/train_array.sbatch
```

`simulate_array.sbatch`, `features_array.sbatch` and
`train_array.sbatch` each dispatch one independent model per array task
so model outputs do not collide. The Slurm account and environment
`PAS2943` / `/users/PAS2197/benson31/yes/bin/python` were inherited
from the original pipeline; adjust them on OSC if needed. Simulator
jobs are CPU-only; train_array requests one GPU per task. Jobs may
require more than 24 hours; check cluster limits and job logs.

For model comparison, the classifier uses the legacy `data2param`
module, while per-model parameter estimation uses `data2param_flow`.
The MR track uses `--for-test` switches separately on
`compare_regen/s1_regen.py`, `compare_regen/s2_getmaxRT.py`,
`compare_regen/s3_fe1_maxT.py` and
`compare_regen/s4_train_classifier_maxT.py`, preserving the distinct
`mr` output directory. `s6_decode_maxT_mr.py` decodes independent
model-recovery test simulations.

## Layout: keep production outputs parallel to food_equal

- `data/krajbich2010/trial_eye.csv`: canonical converted observational input
- `outputs/krajbich2010/tests/`: historical s0 diagnostics and s1–s5 exploratory prior tests
- `outputs/krajbich2010/s1_gen1/`, `s3_fe1/`, `s3_fe1_maxT/`, `dpsRH1_dp0.15/`, `compare_regen/`: same top-level production layout as `food_equal`
- `outputs/krajbich2010/s1_gen1_test/` and `s3_fe1_maxT/test/`: production model-recovery test sets, matching the food_equal layout (not the historical calibration tests)

Migration from the previous layout (perform only when simulation jobs are not reading the old input): move the former s0 trial table into `data/krajbich2010/trial_eye.csv`, and move the s0 diagnostics and all standalone s1–s5 prior diagnostic folders into `outputs/krajbich2010/tests/`. No existing training or model recovery outputs should be moved. Rerun `preflight.py` after migration.
