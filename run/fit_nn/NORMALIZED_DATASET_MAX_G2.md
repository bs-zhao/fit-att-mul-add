# Dataset-max normalized values: matched-prior BayesFlow comparison (2026-11-16)

## Experimental definition

Run two datasets independently: `food_equal` and `krajbich2010`.
Within each dataset, determine one denominator
`M = max(v0, v1)` **over all finite option values after the legacy
fixation-array loading filter** in `data/<dataset>/trial_eye.csv`.
Apply `v0' = v0 / M`, `v1' = v1 / M` before making any empirical
simulation design or encoding a real observation. The maximum of the
loaded, scaled dataset must equal 1. Do **not** normalize independently
per participant, option, pair, or trial. A negative raw value, if present,
remains negative after scaling; this is divide-by-max, not min-max mapping.

**The Cython mathematical simulators are unchanged**. They receive the
normalized values from `Experiment._designs` in S4/S5/cross. The same
normalization applies to `Experiment.groups` in S6. The first two
BayesFlow feature SCALE factors become 1 (not 10), so normalized values
are not divided by 10 again.

The new scripts set `ATT_BF_VALUE_NORMALIZATION=dataset_max` before
importing data. Historical scripts without that flag retain their
unchanged raw-value behavior. No source CSV is rewritten.

## All-model matched priors

| Parameter | Prior in both datasets and all applicable models |
|---|---|
| d | Uniform(1, 50) |
| a | Uniform(0.5, 5) |
| theta (Mul) | Uniform(0, 1) |
| gamma (Add) | Uniform(0, 2) |

The 10-fold increase in the d range is a **provisional compensation**
for scaling values down by a factor on the order of 10, rather than a
validated or uniquely appropriate prior. Evidence thresholds a remain
in accumulator units; theta is dimensionless. Gamma remains additive in
**normalized value units** and its new maximum of 2 may exceed the
maximum normalized value 1. Test prior predictive behavior and S5
recovery before treating any real-data result as definitive.

The four-model class priors remain uniform (0.25 each); separate
four-DDM and four-RACE classifiers are independently trained per
dataset. Training architecture, stopping schedule, 1-ms integration,
original empirical fixation schedules, and maximum RT constraints are
unchanged.

## Code

- `run/fit_nn/<dataset>_bf/_shared/data.py`: opt-in normalization
- `run/fit_nn/<dataset>_bf/experiments/normalized_value_prior_sweep.py`: S4/S5/S6/cross/report
- `run/fit_nn/<dataset>_bf/experiments/parallel_normalized_value_prior_sweep.py`: 8-worker S4

Each `prior_spec.json` stores the input SHA256, the **observed raw
denominator M**, raw minimum, feature normalization, and full model
priors. Food Equal also stores normalization mode in the BayesFlow
checkpoint metadata prior spec; Krajbich stores its specialized prior
spec including the exact normalization audit. S5/S6 cross-check the
saved specification before loading the checkpoint. Old checkpoints
cannot be used by the new experiment.

## Commands (from the repository root)

```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
git pull --ff-only

for dataset in food_equal krajbich2010; do
  for family in ddm race; do
    sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=9 \
      --time=100:00:00 --mem=32G --job-name="${dataset}-N2-${family}" \
      --wrap="cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/run/fit_nn/${dataset}_bf && \
      export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
      python -u experiments/parallel_normalized_value_prior_sweep.py \
      --sim-workers 8 --stage s4 --family ${family} --mode formal && \
      python -u experiments/normalized_value_prior_sweep.py \
      --stage s6 --family ${family} --mode formal"
  done
done
```

Result root for each dataset:

```text
outputs/<dataset>_bf/prior_sensitivity/normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0/
```

To evaluate S5 on the newly trained classifiers:

```bash
python -u experiments/normalized_value_prior_sweep.py --stage s5 --family ddm --mode formal --n-per-model 1000
python -u experiments/normalized_value_prior_sweep.py --stage s5 --family race --mode formal --n-per-model 1000
```

Cross-architecture RACE-Add generated data after **both** four-model
classifiers in a dataset have finished:

```bash
python -u experiments/normalized_value_prior_sweep.py --stage cross --mode formal --generator aRACE_g --n 1000
```

## Comparison and caveats

Use the pairwise `log BF(Mul/Add)` and
`P(Mul>Add) = # subjects with positive pairwise log BF / N` independently
for DDM and RACE. Also report 4-model Top-1 and mean posterior model
probabilities. Do not equate Top-1 differences with pairwise BF evidence.
The experiments intentionally change both the input value scale and
the prior distribution relative to the historical G12 runs; do not
attribute differences solely to gamma. The selected prior must also
be evaluated by S5 recovery, calibration and prior-predictive checks.
