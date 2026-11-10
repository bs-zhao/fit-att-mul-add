# Eight-model shared-prior sensitivity sweep

## Goal

An **isolated**, parameter-prior sensitivity check within the original
Food Equal BayesFlow 2.1.0 code path. All eight models ALWAYS use identical
numerical intervals for the same free parameter:

- All eight: d ~ U(d_min, d_max)
- All eight: a ~ U(a_min, a_max)
- Both Mul models: theta ~ U(0, 1); other models keep theta fixed.
- Both Add models: gamma ~ U(0, 5); other models keep gamma fixed.
- All remaining fixed parameters and model equations are unchanged.

The initial NEW single-factor condition is **d ~ U(0.1,5), a ~ U(1,5)**.
Compare it with the already existing d ~ U(0.1,5), a ~ U(0.5,5)
checkpoint: changing only a_min isolates the effect of its lower bound.

Do **not** interpret a configuration producing the desired classification
as proof that RACE-Add fits the data better; document all tested configurations,
and use model-recovery/calibration and an independent predictive assessment.

## Entrypoint

Run from run/fit_nn/food_equal_bf:

```bash
export KERAS_BACKEND=torch
python -m unittest discover -s tests -p test_shared_prior_sweep.py -v

python -u experiments/shared_prior_sweep.py --stage s4 --family all --mode formal \
  --d-min 0.1 --d-max 5 --a-min 1 --a-max 5
```

The four prior switches are OPTIONAL. Defaults are
`--d-min 0.1 --d-max 5 --a-min 1 --a-max 5`. Use exactly the same
switches for S4, S5, S6, cross and report.

Every prior combination gets its own directory under:

```text
outputs/food_equal_bf/prior_sensitivity/
  sweep_v1_shared_d0p1to5p0_a1p0to5p0_theta0to1_gamma0to5/
    classifier/
      aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/
      aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/
      formal/                     # all eight
    cross_architecture/
```

The prefix `sweep_v1_shared` is deliberately new so even identical
ranges to previous experiments are under **separate paths**. A second
S4 command for the same profile refuses to overwrite an existing
`best.keras` unless `--overwrite` is explicitly provided. Avoid
`--overwrite` when maintaining prior experiment archives.

### Full stages (example of default configuration)

S4 training (three independent jobs):

```bash
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=FE-SWP-DDM-S4 --output=FE-SWP-DDM-S4-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_prior_sweep.py --stage s4 --family ddm --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=FE-SWP-RACE-S4 --output=FE-SWP-RACE-S4-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_prior_sweep.py --stage s4 --family race --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=FE-SWP-ALL-S4 --output=FE-SWP-ALL-S4-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_prior_sweep.py --stage s4 --family all --mode formal"
```

S5 held-out model recovery, AFTER the corresponding S4 finishes:

```bash
python -u experiments/shared_prior_sweep.py --stage s5 --family ddm --mode formal --n-per-model 1000
python -u experiments/shared_prior_sweep.py --stage s5 --family race --mode formal --n-per-model 1000
python -u experiments/shared_prior_sweep.py --stage s5 --family all --mode formal --n-per-model 1000
```

S6 real participant inference, AFTER the corresponding S4 finishes:

```bash
python -u experiments/shared_prior_sweep.py --stage s6 --family ddm --mode formal
python -u experiments/shared_prior_sweep.py --stage s6 --family race --mode formal
python -u experiments/shared_prior_sweep.py --stage s6 --family all --mode formal
```

RACE-Add generated --> 4-DDM vs 4-RACE, AFTER both four-model S4 stages:

```bash
python -u experiments/shared_prior_sweep.py --stage cross --generator aRACE_g \
  --mode formal --seed 20261102 --n 1000 --batch-size 25
```

Summary plots for four-model S5/S6, AFTER all these stages finish:

```bash
python -u experiments/shared_prior_sweep.py --stage report --family both --mode formal
```

For a subsequent different shared prior range, for example d ~ U(.1,6),
a ~ U(.5,5), append `--d-min 0.1 --d-max 6 --a-min 0.5 --a-max 5`
to EVERY stage. The separate output root is computed automatically.

## Provenance and compatibility

- New code only: `experiments/shared_prior_sweep.py`,
  `tests/test_shared_prior_sweep.py`, and this README.
- S4 writes `prior_spec.json` and `metadata.json` with all free priors;
  S5/S6/cross reject checkpoints with incompatible priors and model names.
- No changes to original `food_equal`, `krajbich2010`,
  `krajbich2010_bf`, existing `shared_da_classifier.py`, existing
  `shared_d8_classifier.py`, or unqualified S1-S6.
- No jobs are launched by this commit; execute on OSC after reviewing tests.
