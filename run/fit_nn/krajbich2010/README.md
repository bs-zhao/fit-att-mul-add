# Krajbich 2010 isolated 8-model SBI pipeline

Mirrors `run/fit_nn/food_equal/` with unchanged model equations and file names.
Data and outputs never overlap food_equal. Execute commands below from repository
root; each entry script sets its own working directory.

## Input and RT protocol

Prepare `outputs/krajbich2010/s0_prepare_data/trial_eye.csv` from
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

All outputs: `outputs/krajbich2010/`. No existing `food_equal`
training files or models are touched.
