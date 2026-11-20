# food_equal: eight-model BayesFlow parameter decoder (S1-S3)

New isolated runner: `run/fit_nn/food_equal_parameter_bf/`. Reuses the existing
`run/fit_nn/food_equal_bf` BayesFlow 2.1.0 implementations,
simulator, Cython equations, 17-feature representation, model metadata,
CouplingFlow spline parameter posterior and the original diagnostic code.

Eight available models:
`aDDM_1`, `aDDM_2`, `aDDM_t`, `aDDM_g`,
`aRACE_1`, `aRACE_2`, `aRACE_t`, `aRACE_g`.

The intended initial two models are **aRACE_g (RACE-Add) and
aDDM_t (DDM-Mul)**. No model comparison is being performed in S1-S3:
each model independently learns its conditional parameter posterior.

**Preserved database-specific priors:** this runner calls existing food_equal_bf
and does not impose the separate Tavares G2 d/a bounds. In food_equal_bf,
theta=U(0,1) and gamma=U(0,5), as specified in its data adapter.
For krajbich2010_bf all parameter bounds come from
`run/fit_nn/krajbich2010/tools/model_info.py`.
The S1 run_manifest.json records the exact free parameter names and bounds.

**Formal S1:** 50 batches/epoch × 64 simulated participants/batch =
3,200 generated participants per epoch, max 5,000 epochs, validation 2,000,
best checkpoint and early stopping (upstream LR-patience 8 and early-patience
20). These are training limits, not convergence guarantees. Training audit
checks the best validation checkpoint and actual epochs.
**S2:** 2,000 freshly simulated participants, 1,000 posterior samples each,
recovery correlation, normalized RMSE, bias, 90% coverage, SBC diagnostics.
**S3:** 2,000 posterior samples for every real participant. PPC disabled
in the first pass to avoid expensive re-simulation; enable separately as
a deliberate additional step if needed.

## OSC submission

```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
git pull --ff-only
cd run/fit_nn/food_equal_parameter_bf/
bash submit_first_two.sh both
```

The command creates **six Slurm jobs for this dataset** (two models ×
S1/S2/S3). Only S1 runs first; S2 waits for S1 success, and S3 waits
for S2 success. To run individual stages interactively inside a GPU
allocation:

```bash
python -u s1.py --model aRACE_g
python -u s2.py --model aRACE_g
python -u s3.py --model aRACE_g --ppc-draws 0
```

Replace aRACE_g with aDDM_t for the second model, or any of the six
remaining model names to enable all eight.

Outputs:
`outputs/food_equal_parameter_bf/v1/<model>/formal/`
- S1: best.keras, final.keras, metadata.json, run_manifest.json,
  training_audit.json, history.csv, simulation_stats.json.
- S2: diagnostics/recovery.npz, parameter_recovery_metrics.csv,
  coverage.csv, sbc_ranks.csv, recovery_sbc.png, training_loss.png.
- S3: real_data/posterior_samples.npz, parameter_estimates.csv,
  trial_audit.csv; PPC outputs only if PPC enabled.

Historical checkpoints are not overwritten. This is separate from
S4 model classification and reports no intermodel Bayes factors.
