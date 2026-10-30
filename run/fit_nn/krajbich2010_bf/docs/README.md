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

Training refuses to overwrite `best.keras` unless `--overwrite` is supplied. Model/mode output trees are separate. Defaults: formal = 5000-epoch ceiling, 50 batches/epoch, batch 64, validation 2000; test = 5 epochs, 2 batches/epoch, batch 8, validation 16 (a workflow smoke test, not a recovery benchmark). Both modes keep the same architecture. `--epochs`, `--batches`, `--batch-size`, `--validation`, `--seed`, `--input`, and `--output` allow controlled runs. Best full models are saved atomically during training; diagnostics can read a completed best checkpoint before training ends.

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
