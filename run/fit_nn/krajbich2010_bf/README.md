# Krajbich 2010 — BayesFlow 2.1.0

Eight-model amortized parameter posteriors and Bayesian model comparison. This is a new, independent output pipeline; it does not overwrite the legacy networks or results.

## What is reused and what changes

- Source: `../krajbich2010/tools/model_info.py` and `funcs.py`, imported by absolute file path. All eight model names, parameter order, uniform prior ranges, and fixed parameters are unchanged. `aRACE_g` retains its original gamma range **[0.1, 1]**, whereas `aDDM_g` uses **[0, 1]**.
- Simulator: the existing compiled `packages/c_bifood` functions, with `dt=.001`, `extend_last=500000`, `repeat=0`, `rcd=1`, `max_rt=28.287`. Values remain `[right, left]`; choice/fixation 1 means left. Simulator inputs are copied.
- Each synthetic participant draws one parameter vector and one empirical participant's value/fixation design. Nonhits are omitted as in the legacy generator. Simulated fixation sequences are the truncated/extended sequences returned by the simulator, not the original observed durations.
- Trial feature order is exactly `trialinfo[2] + choice_one_hot[2] + rt[1] + fsmr[12]` (17 dimensions). See `data.FEATURE_NAMES`. A fixed, documented invertible division by `data.SCALE` improves numerical conditioning. Both simulated and observed data use the same encoding and old maxT/maxNF rules.
- Variable participant trial counts use zero padding plus BayesFlow 2.1's `summary_mask`; padding is excluded from SetTransformer attention/pooling. `log(1+n_retained)` is a direct condition, because posterior uncertainty and model evidence depend on sample size. Mask and padded rows are never standardized.
- The reference architecture is retained: SetTransformer(summary_dim=32), spline CouplingFlow, Adam(5e-4, clipnorm=1.5), ReduceLROnPlateau and EarlyStopping. Parameter adapters constrain samples to each original prior interval. Classification uses BayesFlow's ModelComparisonWorkflow with cross-entropy and a uniform eight-model training prior.
- Fresh online simulations replace the legacy pickle/data2param training engine. No `data2param` or `data2param_flow` dependency is required, and no posterior-based regeneration is used for model comparison. This follows the prior-predictive logic of `compare_noregen` and the reference project.

Reference project reviewed: `bs-zhao/intertemporal-bf-1` at `6c5d66cb5dd711a64a681d3cb6ae02e47265b716`. Legacy source reviewed at `a5cd9df9e564489a8b6fcacd00a47988f592c08e`. This implementation uses **2.1.0 estimator APIs** (`workflow.estimator`), not the reference's 2.0.13 serialization.

## Installation

BayesFlow 2.1.0 (latest PyPI release checked 2026-10-30; released 2026-10-23) requires Python 3.12 or 3.13. Select an environment with a suitable GPU PyTorch installation first. An existing Python 3.11 environment cannot be reused.

From the repository root:

```bash
# Install torch appropriate to your platform / CUDA version first.
python -m pip install torch
bash run/fit_nn/krajbich2010_bf/setup.sh
```

On OSC, activate a Python 3.12/3.13 environment with GPU PyTorch, then run `setup.sh`. The supplied Slurm wrapper inherits this environment; it deliberately does not guess an OSC module or activate the old Python 3.11 environment. Submit it from this folder. If your allocation uses a different account, supply `sbatch --account=YOUR_ACCOUNT ...`.

Required input (not tracked in Git): `data/krajbich2010/trial_eye.csv`. Its schema is the same as the legacy loader (`subj,v0,v1,response,arr_ml3_left,arr_ml3_time`). RT uses Scheme A: sum of fixation durations, in seconds; `rt_original_ms` is not used for fitting. The input SHA256 is recorded and checked before diagnostics/decoding to prevent accidental design drift.

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

Or run every stage for all eight models sequentially:

```bash
python run_all.py --mode test
python run_all.py --mode formal
```

On OSC:

```bash
sbatch run.sbatch run_all.py --mode test
# After inspecting smoke-test outputs:
sbatch run.sbatch run_all.py --mode formal
# Independent jobs are possible for each parameter network:
sbatch run.sbatch s1_train_parameter_posterior.py --model aRACE_g --mode formal
sbatch run.sbatch s4_train_model_classifier.py --mode formal
```

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
