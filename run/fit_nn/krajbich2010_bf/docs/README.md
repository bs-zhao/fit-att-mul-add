# Krajbich 2010 — BayesFlow 2.1.0

Eight-model amortized parameter posteriors and Bayesian model comparison. This is a new, independent output pipeline; it does not overwrite the legacy networks or results.

## What is reused and what changes

- Source: `../krajbich2010/tools/model_info.py` and `funcs.py`, imported by absolute file path. All eight model names, parameter order, uniform prior ranges, and fixed parameters are unchanged. `aRACE_g` retains its original gamma range **[0.1, 1]**, whereas `aDDM_g` uses **[0, 1]**.
- Simulator: the existing compiled `packages/c_bifood` functions, with `dt=.001`, `extend_last=500000`, `repeat=0`, `rcd=1`, `max_rt=28.287`. Values remain `[right, left]`; choice/fixation 1 means left. Simulator inputs are copied.
- Each synthetic participant draws one parameter vector and one empirical participant's value/fixation design. Nonhits are omitted as in the legacy generator. Simulated fixation sequences are the truncated/extended sequences returned by the simulator, not the original observed durations.
- Trial feature order is exactly `trialinfo[2] + choice_one_hot[2] + rt[1] + fsmr[12]` (17 dimensions). See `_shared.data.FEATURE_NAMES`. A fixed, documented invertible division by `_shared.data.SCALE` improves numerical conditioning. Both simulated and observed data use the same encoding and old maxT/maxNF rules.
- Variable participant trial counts use zero padding plus BayesFlow 2.1's `summary_mask`; padding is excluded from SetTransformer attention/pooling. `log(1+n_retained)` is a direct condition, because posterior uncertainty and model evidence depend on sample size. Mask and padded rows are never standardized.
- The reference architecture is retained: SetTransformer(summary_dim=32), spline CouplingFlow, Adam(5e-4, clipnorm=1.5), ReduceLROnPlateau and EarlyStopping. Parameter adapters constrain samples to each original prior interval. Classification uses BayesFlow's ModelComparisonWorkflow with cross-entropy and a uniform eight-model training prior.
- Fresh online simulations replace the legacy pickle/data2param training engine. No `data2param` or `data2param_flow` dependency is required, and no posterior-based regeneration is used for model comparison. This follows the prior-predictive logic of `compare_noregen` and the reference project.

Reference project reviewed: `bs-zhao/intertemporal-bf-1` at `6c5d66cb5dd711a64a681d3cb6ae02e47265b716`. Legacy source reviewed at `a5cd9df9e564489a8b6fcacd00a47988f592c08e`. This implementation uses **2.1.0 estimator APIs** (`workflow.estimator`), not the reference's 2.0.13 serialization.

## Folder layout and installation

The top level contains only the six `s1`–`s6` Python entrypoints plus organized subfolders:

- `_shared/`: shared simulation, encoding, workflow and diagnostic code.
- `config/requirements.txt`: pinned BayesFlow dependency.
- `docs/`: this guide and validation record.
- `tests/`: feature, mask, simulator, training and serialization checks.

BayesFlow 2.1.0 requires Python 3.12 or 3.13. Activate your GPU-enabled PyTorch environment before submitting jobs. Install dependencies once from the repository root:

```bash
python -m pip install torch
python -m pip install -r run/fit_nn/krajbich2010_bf/config/requirements.txt
python -m pip install --no-build-isolation -e packages/c_bifood
```

There is no separate shell/Slurm wrapper. Submit the Python stages directly using `sbatch --wrap`, as shown below. The job inherits your activated environment. Keep the entire `_shared` folder alongside the six scripts.

Required input (not tracked in Git): `data/krajbich2010/trial_eye.csv`. Its schema is the same as the legacy loader (`subj,v0,v1,response,arr_ml3_left,arr_ml3_time`). RT uses Scheme A: sum of fixation durations, in seconds; `rt_original_ms` is not used for fitting. The input SHA256 is recorded and checked before diagnostics/decoding to prevent accidental design drift.

## S4 classifier training defaults (formal mode)

S4 now uses **batch_size=256**, **30 batches per epoch**,
**ReduceLROnPlateau(patience=4)** and **EarlyStopping(patience=10)**.
Its Adam learning rate is still 5e-4, with the same validation set size
(2,000 simulated participants) and early stopping min_delta (1e-4).
S1 parameter posterior training stays at batch_size=64 and the original
patience values (8 for LR reduction, 20 for early stopping).
The S4 `--mode test` batch size remains 8; CLI `--batch-size`
still overrides the default in either mode.

With 30 batches per epoch, S4 now simulates **7,680 participants/epoch**,
compared with 3,200 in the original S4 (batch 64, 50 batches). This may INCREASE epoch time: online simulation
is performed anew every batch. To keep the number of simulated participants
near the previous level with batch_size=256, use `--batches 13` (3,328
participants/epoch). Changing batch size alone is not a speed optimization.

```bash
python s4_train_model_classifier.py --mode formal --models aDDM_t aDDM_g
# Or keep approximately the original number of simulated participants/epoch:
python s4_train_model_classifier.py --mode formal --models aDDM_t aDDM_g --batches 13
```

A currently running Slurm job continues using the values loaded when it
started. Changing these settings requires starting a new S4 training job,
and an existing `best.keras` checkpoint must be intentionally handled via
`--overwrite` (which deletes that checkpoint) or by backing it up first.

## Arbitrary 2-8 model subsets for S4-S6

Use `--models MODEL [MODEL ...]` on **each** of S4, S5, and S6 to select any
two to eight distinct candidates. Available names are `aDDM_1`, `aDDM_2`,
`aDDM_t`, `aDDM_g`, `aRACE_1`, `aRACE_2`, `aRACE_t`, `aRACE_g`.
All three stages must use the same set. Names are ordered canonically as
listed here regardless of argument order; model columns and `--model-prior`
values follow that order.

```bash
# Binary comparison, with independent training and diagnostics
python s4_train_model_classifier.py --mode formal --models aDDM_t aDDM_g
python s5_model_recovery.py --mode formal --models aDDM_t aDDM_g
python s6_classify_real_data.py --mode formal --models aDDM_t aDDM_g

# Four-model classifier, separate checkpoint
python s4_train_model_classifier.py --mode formal --models aDDM_t aDDM_g aRACE_t aRACE_g

# Optional two-model prior (canonical order)
python s6_classify_real_data.py --mode formal --models aDDM_t aDDM_g --model-prior 0.3 0.7
```

For proper subsets, output is isolated under
`outputs/krajbich2010_bf/classifier/<canonical_model_names_joined_with_double_underscores>/<mode>/`.
For example, the binary comparison saves to
`outputs/krajbich2010_bf/classifier/aDDM_t__aDDM_g/formal/`.
The all-eight default, including an explicit full selection, continues to use
the existing `classifier/formal/` tree, preserving older checkpoints.
Use `--overwrite` to retrain a previously saved subset.

S4 trains one **K-class** classifier (K=2..8) from fresh prior-predictive
simulations, independently of S1-S3. S5 performs K-class model recovery with
K×K confusion and mean-probability matrices, using `--n-per-model` datasets
*per included model*. S6 exports K posterior probabilities and K(K-1)
ordered pairwise log Bayes factors per participant. `--model-prior` accepts
exactly K positive probabilities summing to 1; its default is uniform.

DDM/ACC family sums and recovery are name-based, not positional. When the
subset contains only one family, its probability and family accuracy are
trivially 1, so family accuracy is not informative. All posterior model
probabilities and Bayes factors are **conditional on the selected model set**.

## Six stages

All Python entrypoints resolve data/output paths from their own location and work from any directory. Below, change into `run/fit_nn/krajbich2010_bf` for shorter commands.

```bash
# 1. Train one parameter posterior (repeat for all eight models)
python s1_train_parameter_posterior.py --model aDDM_t --mode test
# 2. Independent parameter recovery, SBC, coverage and posterior dependence
python s2_parameter_recovery.py --model aDDM_t --mode test --draws 100
# 3. All observed participants: posterior samples, estimates, posterior predictive checks
python s3_decode_real_data.py --model aDDM_t --mode test --draws 100 --ppc-draws 5
# 4. Train the eight-class model comparison estimator
python s4_train_model_classifier.py --mode test
# 5. Balanced independent model recovery and probability calibration
python s5_model_recovery.py --mode test
# 6. All observed participants: PMPs and pairwise log Bayes factors
python s6_classify_real_data.py --mode test
```

For OSC, change into `run/fit_nn/krajbich2010_bf`, activate your environment, and submit each stage directly. Formal examples:

```bash
sbatch --account=PDE0114 --gres=gpu:1 --time=100:00:00 --mem=32G --wrap="python -u s1_train_parameter_posterior.py --model aDDM_t --mode formal"
sbatch --account=PDE0114 --gres=gpu:1 --time=100:00:00 --mem=32G --wrap="python -u s2_parameter_recovery.py --model aDDM_t --mode formal"
sbatch --account=PDE0114 --gres=gpu:1 --time=100:00:00 --mem=32G --wrap="python -u s3_decode_real_data.py --model aDDM_t --mode formal"
sbatch --account=PDE0114 --gres=gpu:1 --time=100:00:00 --mem=32G --wrap="python -u s4_train_model_classifier.py --mode formal"
sbatch --account=PDE0114 --gres=gpu:1 --time=100:00:00 --mem=32G --wrap="python -u s5_model_recovery.py --mode formal"
sbatch --account=PDE0114 --gres=gpu:1 --time=100:00:00 --mem=32G --wrap="python -u s6_classify_real_data.py --mode formal"
```

These are separate submissions, not an automatic dependency chain. Wait for step 1 to finish before steps 2/3, and for step 4 before steps 5/6. Repeat steps 1–3 for each of the eight `--model` names. Step 4 trains one classifier covering all eight models and can run independently of the parameter networks. First use `--mode test` for a short smoke run, then switch to `--mode formal`.

Training refuses to overwrite `best.keras` unless `--overwrite` is supplied. Model/mode output trees are separate. Defaults: formal S1 = 5000-epoch ceiling, 50 batches/epoch, batch 64; formal S4 = 5000-epoch ceiling, 30 batches/epoch, batch 256; both use validation 2000. Test = 5 epochs, 2 batches/epoch, batch 8, validation 16 (a workflow smoke test, not a recovery benchmark). Both modes keep the same architecture. `--epochs`, `--batches`, `--batch-size`, `--validation`, `--seed`, `--input`, and `--output` allow controlled runs. Best full models are saved atomically during training; diagnostics can read a completed best checkpoint before training ends.

## Additional experiment: shared RACE-Add simulated data across DDM and RACE

Use aRACE_g to independently generate synthetic participants from its original
uniform (d, a, gamma) priors. The **same** 17-dimensional data, fixation mask,
and retained-trial count are passed to existing four-DDM and four-RACE
classifiers (both loaded from best.keras). This does NOT train new networks.

Run from the krajbich2010_bf folder:

```bash
python -u experiments/race_add_cross_architecture.py --mode formal --n 1000 --batch-size 25
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=1 --time=100:00:00 --mem=32G --job-name=bf-RACEadd-cross --output=slurm-cross-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/race_add_cross_architecture.py --mode formal --n 1000 --batch-size 25"
```

Required checkpoints: classifier/aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/best.keras
and classifier/aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/best.keras.
Output directory: outputs/krajbich2010_bf/cross_architecture/generated_aRACE_g/formal/seed_20261102/.
Outputs include per_simulation.csv, summary.json, gamma_quartiles.csv, and
paired log-BF scatter, delta histogram, gamma dependence and PMP plots.

The paired estimand is DDM log BF(Mul/Add) minus RACE log BF(Mul/Add),
both computed within each four-model classifier with 0.25 model prior.
Positive delta means DDM shows stronger *false* support for Mul given
RACE-Add-generated data. Paired t tests (two-sided and directional),
Wilcoxon and paired bootstrap CI are reported. Because classifier model
sets differ, this is a cross-family misspecification stress test, not a
direct Bayes factor between diffusion and race architectures.
### Symmetric experiment: RACE-Mul as the data generator

The same diagnostic now accepts `--generator aRACE_g` (original default)
or `--generator aRACE_t`. All existing RACE-Add commands and paths are
unchanged. The RACE-Mul wrapper defaults to `aRACE_t` so it is easy to run
as a separate Slurm job:

```bash
python -u experiments/race_mul_cross_architecture.py --mode formal --n 1000 --batch-size 25
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=1 --time=100:00:00 --mem=32G --job-name=bf-RACEmul-cross --output=slurm-crossmul-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/race_mul_cross_architecture.py --mode formal --n 1000 --batch-size 25"
```

Generated data follow the original `aRACE_t` priors for `d`, `a`, and
`theta`, and are identical between both classifier evaluations. A positive
paired delta means stronger Mul support from DDM than RACE. Under the Mul
generator, this is not a false-positive measure; compare it with the
RACE-Add case to distinguish a general multiplicative tilt from selective
failure to identify additive-generated data.

Outputs are independent of the earlier experiment:
`outputs/krajbich2010_bf/cross_architecture/generated_aRACE_t/formal/seed_20261102/`.
The result files are `per_simulation.csv`, `summary.json`,
`theta_quartiles.csv` and plots (`paired_delta_vs_theta.png`, etc.).
Original Add results continue to use `gamma_quartiles.csv`.

## Wide-prior classifier sensitivity analysis (gamma U(0,5), theta U(0,1))

The baseline tools/model_info.py and existing S4/S5/S6 classifiers remain unchanged.
A separate experiment wrapper overrides only in-memory free parameter ranges
for RACE/DDM Add gamma and RACE/DDM Mul theta, then calls the original S4/S5/S6.
All four-model class priors remain 0.25. Original unfiltered input is reused.

Run from run/fit_nn/krajbich2010_bf:

```bash
git pull --ff-only
sbatch --account=PAS2943 --gres=gpu:1 --time=100:00:00 --mem=32G --job-name=wide-RACE --output=wide-RACE-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/wide_prior_classifier.py --stage s4 --family race --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --time=100:00:00 --mem=32G --job-name=wide-DDM --output=wide-DDM-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/wide_prior_classifier.py --stage s4 --family ddm --mode formal"
```

After S4 finishes successfully, run S5 held-out recovery and S6 real data:

```bash
python -u experiments/wide_prior_classifier.py --stage s5 --family race --mode formal
python -u experiments/wide_prior_classifier.py --stage s6 --family race --mode formal
python -u experiments/wide_prior_classifier.py --stage s5 --family ddm --mode formal
python -u experiments/wide_prior_classifier.py --stage s6 --family ddm --mode formal
```

Output base: outputs/krajbich2010_bf/prior_sensitivity/gamma_0_5_theta_0_1/.
Each classifier subset/mode has its own prior_spec.json, best.keras,
simulation_stats.json, diagnostics and real_data. New priors are used for
both S4 training and S5 model recovery. S6 uses the matched new best.keras.
Train refuses to overwrite existing checkpoints unless --overwrite is explicit.
Test mode: --mode test --epochs 2 --batches 2 --validation 16 for S4;
test mode S5: --mode test --n-per-model 12.
Formal mode retains original training defaults: 5000 epoch ceiling,
30 batches per epoch, batch size 256, validation 2000 with early stopping.

Because prior predictive distributions and Occam factors change, a wider
gamma prior need not increase Add model evidence. Inspect S5 calibration and
real-data log BF(Mul/Add) before interpreting any shift.
## Exact-parity S4 simulation acceleration

`_shared/data.py` now uses Cython `rcd=0` to avoid accumulating and
materializing per-millisecond latent traces that S4 never consumes.
The returned choice/RT are unchanged; the observed fixation schedule
is truncated deterministically at the simulated 1ms decision step,
including the simulator's original 500000ms final-fixation extension.
Fixation summary features are computed from durations directly, merging
adjacent identical gaze segments exactly as `get_summary(arr2tp(...))` does.
Empirical value/fixation arrays are cached per participant for reuse.
No change to prior distributions, model formulas, dt, dataset sizes,
network architecture, or optimization hyperparameters.

Verify bitwise equality of the 17 float32 features under identical
NumPy and libc RNG seeds **before** launching expensive S4 runs:

```bash
python -u experiments/verify_fast_simulation.py --family race --n-subjects 2
python -u experiments/verify_fast_simulation.py --family ddm --n-subjects 2
```

Run formal new-prior S4 with automatic preflight (separate RACE / DDM jobs):

```bash
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=wide-RACE-fast --output=wide-RACE-fast-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/verify_fast_simulation.py --family race --n-subjects 2 && python -u experiments/wide_prior_classifier.py --stage s4 --family race --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=wide-DDM-fast --output=wide-DDM-fast-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/verify_fast_simulation.py --family ddm --n-subjects 2 && python -u experiments/wide_prior_classifier.py --stage s4 --family ddm --mode formal"
```

Preflight fails the job before training on any parity regression. Reported
simulation-only speedups are local benchmarks, not a guaranteed end-to-end
training speedup. Existing checkpoint paths remain unchanged; S4 refuses
to overwrite an existing best.keras unless --overwrite is supplied.
## Complete current-prior S5/S6 report figures

Current-prior reports are generated automatically when both S5 and S6 have
completed for a classifier. When both the four-RACE and four-DDM pipelines
are complete, their 2x4 combined plots are generated as well. The report
contains only the current classifier results; figures have no historical
prior overlays or old/new labels.

To regenerate all plots from **existing outputs** without retraining or
recomputing model posteriors:

    python -u experiments/wide_prior_classifier.py --stage report --family both --mode formal

For just one classifier replace --family both with race or ddm.

Files are saved under:
    outputs/krajbich2010_bf/prior_sensitivity/gamma_0_5_theta_0_1/reports/formal/

Subfolders are race/, ddm/, and combined/. Each family outputs 16 PNG plus
16 vector PDF figures, a summary JSON, subject/model CSV summaries and all
pairwise log-BF/relative-winning-rate CSV matrices. The combined report
outputs 15 PNG plus 15 PDF figures, a paired subject CSV and summary JSON.
All 47 individual charts are available in both PNG and PDF.

Content: four-model winning counts and mean model posterior probabilities,
subject-level probability heatmaps, log BF(Mul/Add) distributions and
individual results, BF strength >10, six pairwise contrasts, entropy and
posterior concentration, S5 confusion, classwise recovery, true-model
probability, calibration (overall and by class), log loss, Brier, ECE,
and RACE/DDM model agreement and cross-architecture differences.

The difference between two within-architecture log BF(Mul/Add) measures
is not a direct Bayes factor between DDM and RACE architectures.
## Outputs

Everything goes to `outputs/krajbich2010_bf` by default:

| Location | Contents |
|---|---|
| `<model>/<mode>/` | `best.keras`, `final.keras`, live `history.csv`, metadata, simulation exclusion counters |
| `<model>/<mode>/diagnostics/` | raw recovery draws/truth, Pearson/Spearman, RMSE/MAE/bias, prior-width normalized RMSE, coverage curve, interval width, contraction, SBC ranks/KS, signed/absolute posterior correlation matrices, figures |
| `<model>/<mode>/real_data/` | identified posterior samples, per-parameter means/medians/SD/90%/95% intervals, trial audit, PPC draws/summaries/plots |
| `classifier/<mode>/diagnostics/` | eight-model confusion matrices, family confusion, mean PMPs, per-model recall, log loss, multiclass Brier score, top-label ECE, classwise reliability, figures |
| `classifier/<mode>/real_data/` | per-participant PMPs, winning model, entropy, family probability sums, pairwise log BFs, heatmap, trial audit |

Formal recovery defaults: 1000 independent datasets per parameter network / per model for classification, 1000 posterior draws, inference batches of 25. PPC defaults to 100 parameter posterior draws per observed participant and compares choice, RT quantiles, gaze and fixation counts. `--ppc-draws 0` skips predictive simulation. PPC is conditional on the empirical value/fixation template and checks the retained-trial distribution generated by this simulator; it is not an unconditional model of human gaze generation.

`s6` accepts `--model-prior` followed by eight positive probabilities in `MODEL_NAMES` order. It removes the uniform **training** prior to compute BFs using BayesFlow's log-odds output, then applies the requested **inference** prior. Output is log BF to avoid numerical overflow; rows explicitly name numerator/denominator. Family probabilities sum member PMPs. There is no claim that a family is a separately fitted model or that mean participant PMPs are a group Bayes factor.

## Interpretation and reproducibility

Model comparison and SBC refer to this simulation/selection procedure, parameter priors, empirical template distribution and retained 17-feature representation. As in the old pipeline, nonhits and maxT violations are omitted and subjects with <=5 retained trials are redrawn. Subject rejection can change the effective prior; inspect `simulation_stats.json`. The uniform-prior contraction statistic uses the nominal prior variance, so it is a descriptive reference if rejection is appreciable. SBC against fresh accepted draws assesses the implemented joint distribution; success does not establish correctness outside that distribution or sufficiency of the trial summaries. KS p-values are descriptive and not multiplicity corrected.

Parameter/design RNG streams differ between training, validation, recovery and PPC. NumPy/Keras and the simulator's libc `rand` are seeded at process entry. Run one simulator process per job (no threaded calls to the global C RNG). Backend/hardware/library changes may change exact draws. Save metadata and trained artifacts together. Native Windows compilation requires a C++ toolchain; Linux/OSC uses the system compiler.

## Verification

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

See `VALIDATION.md` for the executed integration checks. Small fixture data used for verification are generated in temporary directories; no participant data or training outputs are committed.

Official references: [PyPI 2.1.0](https://pypi.org/project/bayesflow/2.1.0/), [BayesFlow source](https://github.com/bayesflow-org/bayesflow/tree/v2.1.0), [workflows](https://bayesflow.org/stable/user_guide/workflows.html).
