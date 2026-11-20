# Unified G2 BayesFlow runs

The six top-level scripts in each of `food_equal_bf`, `krajbich2010_bf`
and `tavares2017_bf` use the same four DDM and four RACE models.
The shared model specification uses `d ~ Uniform(1, 50)`,
`a ~ Uniform(0.5, 5)`, `theta ~ Uniform(0, 1)` for Mul and
`gamma ~ Uniform(0, 2)` for Add. All models fix `ndt = 0` and
`s = 1` and use the same Cython simulator with a 1 ms time step.

Each adapter divides both value columns by a single maximum across its entire
loaded dataset, before constructing simulation designs or real-data features.
The same denominator applies to all participants and both options. Negative
values keep their sign.

| Pipeline | Default input |
| --- | --- |
| Food Equal | `data/food_equal/trial_eye.csv` |
| Krajbich2010 | `data/krajbich2010/trial_eye.csv` |
| Tavares2017 | `data/tavares2017_final/trial_eye.csv` |

Prepare the Tavares input with
`python run/fit_nn/tavares2017_bf/final_preprocessing/prepare_final.py`
from the repository root. See
[the preprocessing notes](tavares2017_bf/final_preprocessing/README.md)
for the input files, off-item fixation QC rule and item-fixation time used as RT.

## S1–S3 parameter estimation

From a dataset's pipeline directory, run the following sequence for each
of the eight model names. These commands use the formal settings by default.

```bash
model=aDDM_t
python -u s1_train_parameter_posterior.py --model "$model"
python -u s2_parameter_recovery.py --model "$model"
python -u s3_decode_real_data.py --model "$model"
```

S2 recovers parameters on fresh simulations. S3 estimates posterior parameters
for the real participants and produces posterior predictive checks.

## S4–S6 model comparison

Run this sequence for the joint eight-model classifier.

```bash
python -u s4_train_model_classifier.py
python -u s5_model_recovery.py
python -u s6_classify_real_data.py
```

Run the same three commands with `--models aDDM_1 aDDM_2 aDDM_t aDDM_g`
for the DDM classifier, or with `--models aRACE_1 aRACE_2 aRACE_t aRACE_g`
for the RACE classifier. S5 diagnoses model recovery and calibration on fresh
simulations. S6 saves participant probabilities and
`real_data/mean_model_probabilities.csv`, the mean of each model's probabilities
across included participants.

Checkpoints and results go to `outputs/<dataset>_bf/unified_g2_v1/`. Use the
same `--mode`, `--input` and `--output` at each stage if you override their
defaults. Checkpoint loading verifies the input file hash, model specification,
raw value maximum and feature scale. Train new S1 and S4 checkpoints for these
runs.
