# Food Equal BayesFlow validation

## Migration status

This directory was copied from `krajbich2010_bf` and retargeted to the
original food_equal model definitions and data. The source implementation
was previously validated for the Krajbich dataset, but those runtime
results are NOT evidence of validation for food_equal.

The following checks should be run on the OSC host with BayesFlow 2.1.0,
the compiled `c_bifood` module, and `data/food_equal/trial_eye.csv`:

```bash
export KERAS_BACKEND=torch
python -m unittest discover -s tests -v
python -u experiments/verify_fast_simulation.py --family race --n-subjects 2
python -u experiments/verify_fast_simulation.py --family ddm --n-subjects 2
python -u s1_train_parameter_posterior.py --mode test --model aRACE_g
python -u s2_parameter_recovery.py --mode test --model aRACE_g --n-test 8 --draws 16
python -u s3_decode_real_data.py --mode test --model aRACE_g --draws 16 --ppc-draws 0
python -u s4_train_model_classifier.py --mode test --models aRACE_1 aRACE_2 aRACE_t aRACE_g
python -u s5_model_recovery.py --mode test --models aRACE_1 aRACE_2 aRACE_t aRACE_g --n-per-model 8
python -u s6_classify_real_data.py --mode test --models aRACE_1 aRACE_2 aRACE_t aRACE_g
```

### Invariants

- `food_equal/tools/model_info.py` is imported and deep-copied; the source
  file and its model definitions are never modified by `food_equal_bf`.
- S1-S6 use theta U(0,1) and gamma U(0,5) (Mul/Add in both families),
  leaving d, a and fixed model parameters untouched.
- Checkpoint metadata contains `prior_spec`. Missing or mismatched
  prior specs cause S2/S3/S5/S6 to refuse loading the checkpoint.
- Synthetic simulator uses `max_rt=14.5` seconds, `dt=0.001`, `extend_last=500000`,
  `repeat=0`, and the same DDM/RACE Cython dynamics.
- Trials are excluded for fixation count >=128 and the original 15,000ms
  downsampled representation threshold.
- The legacy 17-feature summary and 0=right / 1=left coding are preserved.
- Input data SHA, model names, prior, and max_trials metadata are checked before decoding.
- New artifacts are isolated under `outputs/food_equal_bf`.

Full integration validation has not been executed on OSC as part of this
GitHub-only migration. Do not interpret static tests as runtime network validation.

## Food Equal off-target fixation codes

Some food_equal empirical gaze sequences include AOI codes other than 0/1.
The original Cython DDM/RACE simulator assigns every code !=1 to its existing
else branch, while the original food_equal `get_summary` excludes non-0/1
codes when calculating gaze summaries (including number of gaze runs).
The migrated optimized feature extractor now matches these two behaviors:
original coded fixations remain unchanged during simulation, but off-target
codes are removed from the gaze-summary calculation. This fixes S4 validation
simulation failures on nonbinary gaze without changing model dynamics.
Run `python -m unittest discover -s tests -v` and both
`experiments/verify_fast_simulation.py --family race|ddm --n-subjects 2`
preflights on OSC before long training. Only `food_equal_bf` was patched.
