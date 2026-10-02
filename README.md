# fit-att-mul-add

Clean reimplementation of the behavioral fitting/model-comparison pipeline for the manuscript on multiplicative vs. additive gaze bias.

The repository keeps the **same eight computational models and the same current food-choice data interface**, but replaces the old point-parameter `data2param` workflow and PDA/BIC model comparison with:

1. prior-predictive simulation under each candidate model;
2. `data2param-flow` conditional-flow parameter decoder training;
3. participant-specific posterior inference;
4. posterior-conditioned resimulation under each fitted model;
5. an 8-class neural model classifier trained on those regenerated datasets;
6. classifier-based model probabilities for the real participants, plus model-recovery diagnostics.

`data/` and `outputs/` are deliberately ignored by git.

## Candidate models

| Manuscript label | Internal name | Free parameters | Fixed gaze parameters |
|---|---|---|---|
| DDM-NoAtt | `aDDM_1` | d, a | theta=1, gamma=0 |
| DDM-AttOnly | `aDDM_2` | d, a | theta=0, gamma=0 |
| DDM-Mul | `aDDM_t` | d, a, theta | gamma=0 |
| DDM-Add | `aDDM_g` | d, a, gamma | theta=1 |
| ACC-NoAtt | `aRACE_1` | d, a | theta=1, gamma=0, b=0 |
| ACC-AttOnly | `aRACE_2` | d, a | theta=0, gamma=0, b=0 |
| ACC-Mul | `aRACE_t` | d, a, theta | gamma=0, b=0 |
| ACC-Add | `aRACE_g` | d, a, gamma | theta=1, b=0 |

The drift/evidence dynamics are the uploaded `c_bifood.pyx` implementation. The historical multiplicative training range `theta in [0.1, 1]` is retained to match the original code. If the exact manuscript nesting claim `theta in [0,1]` is desired, change the two multiplicative theta ranges in `src/fit_att_mul_add/model_info.py` before generating simulations and retrain from scratch.

## Current dataset

Place the current preprocessed behavioral/eye-tracking file at:

```text
data/trial_eye.csv
```

Required columns are exactly those used by the previous fitting code:

```text
subj, v0, v1, response, arr_ml3_left, arr_ml3_time
```

The historical convention is retained: simulator values are `[v0, v1] = [right, left]`, fixation code `0=right`, `1=left`, and response is binary `0/1`.

## Installation

```bash
git clone https://github.com/bs-zhao/fit-att-mul-add.git
cd fit-att-mul-add
python -m venv .venv
source .venv/bin/activate
pip install -e .
python setup_simulator.py build_ext --inplace
```

`pip install -e .` installs `data2param-flow` directly from:

```text
https://github.com/Cognition-Decision-Modeling-Lab/data2param-flow
```

The relevant API is the revised conditional-flow `ParameterDecoder`: regression is trained by conditional negative log likelihood; `sample_posterior` supplies posterior draws used for regeneration.

## Pipeline

### 0. Verify current data

```bash
python scripts/00_check_data.py
```

### 1. Generate prior-predictive training simulations

For a quick test:

```bash
python scripts/01_simulate_training.py --model aRACE_t --gen-start 1 --gen-end 2 --subjects-per-file 50
```

Full default generation is 50 files x 500 synthetic participants per model:

```bash
python scripts/01_simulate_training.py --model all
```

Each synthetic participant borrows the real trial design (values and observed gaze sequence) from one real participant, samples a parameter vector from the model prior range, and simulates choice/RT using the original `c_bifood` model. As in the submitted likelihood code, the final observed fixation is held when a simulated decision outlasts the observed fixation sequence.

### 2. Build flow-ready features

```bash
python scripts/02_build_training_features.py --model all
```

The default trial representation is shared across all stages:

```text
trialinfo = [v0, v1]
choice    = 2-way one-hot
rt        = response time in seconds
fsmr      = fixation summary (time, proportion, counts, first/last fixation)
```

### 3. Train one `data2param-flow` parameter decoder per model

```bash
python scripts/03_train_parameter_decoder.py --model aRACE_t
```

or run all eight sequentially:

```bash
python scripts/03_train_parameter_decoder.py --model all
```

Models are saved under:

```text
outputs/param_decoder/<model>/parameter_decoder.pkl
```

### 4. Build model-independent features for the real participants

```bash
python scripts/04_build_real_features.py
```

This creates one feature pickle per real participant plus `outputs/real_features/manifest.csv`.

### 5. Infer participant-specific parameter posteriors

```bash
python scripts/05_infer_parameters.py --model all --posterior-samples 500
```

Outputs include posterior centers and posterior draws in native parameter units. A quick parameter-recovery diagnostic is available with:

```bash
python scripts/05b_parameter_recovery.py --model aRACE_t
```

### 6. Posterior-conditioned resimulation

```bash
python scripts/06_resimulate_posterior.py --model all
```

For every regenerated synthetic participant, the code:

- selects a real participant/template;
- samples one draw from that participant's fitted flow posterior under the candidate model;
- preserves that participant's option values and exogenous gaze sequences;
- resimulates choice/RT from the candidate model.

This is the new model-comparison training distribution and replaces PDA/BIC as the primary comparison route.

### 7. Build classifier features

```bash
python scripts/07_build_classifier_features.py --model all
```

### 8. Train the 8-class model classifier

```bash
python scripts/08_train_model_classifier.py
```

The classifier uses the same `data2param-flow` package in `decoder_type="classify"` mode, which retains the data2param classifier architecture/API.

### 9. Model-recovery check

```bash
python scripts/10_model_recovery.py --max-files-per-model 5
```

Inspect the confusion matrix before interpreting real-data model probabilities. A model comparison is not trustworthy if the relevant candidate models cannot be recovered from their own posterior-predictive simulations.

### 10. Decode the real data

```bash
python scripts/09_decode_real_data.py
python scripts/11_summarize_comparison.py
```

Primary outputs:

```text
outputs/model_comparison/real_model_probabilities.csv
outputs/model_comparison/summary.csv
outputs/model_comparison/recovery_counts.csv
outputs/model_comparison/recovery_row_proportions.csv
```

## OSC examples

Parameter decoders are independent and should normally be trained as separate GPU jobs:

```bash
sbatch slurm/train_param.sbatch aDDM_1
sbatch slurm/train_param.sbatch aDDM_2
sbatch slurm/train_param.sbatch aDDM_t
sbatch slurm/train_param.sbatch aDDM_g
sbatch slurm/train_param.sbatch aRACE_1
sbatch slurm/train_param.sbatch aRACE_2
sbatch slurm/train_param.sbatch aRACE_t
sbatch slurm/train_param.sbatch aRACE_g
```

CPU simulation can also be split by generation range, for example:

```bash
sbatch slurm/sim_training.sbatch aRACE_t 1 10
sbatch slurm/sim_training.sbatch aRACE_t 11 20
```

## Important methodological distinction from the rejected submission

The previous reported BIC values evaluated a PDA likelihood after amortized parameter fitting rather than directly optimizing that PDA likelihood. That does **not** guarantee the nested-model likelihood ordering required by an MLE-based BIC comparison. This repository therefore does not reuse those BIC values. The primary comparison is based on classifier discrimination among posterior-conditioned regenerated datasets, with explicit model-recovery diagnostics.

If PDA/BIC is reintroduced later, it should be a separate validation path with direct likelihood optimization (or another method that genuinely targets the likelihood maximum), not simply PDA evaluation at the flow posterior center.
