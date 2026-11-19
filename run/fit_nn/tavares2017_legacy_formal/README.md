# Tavares2017 original-data G2: frozen-model formal retraining

New isolated path: run/fit_nn/tavares2017_legacy_formal.

This deliberately preserves the **historical Tavares2017 original/no-50%-QC dataset**
and all eight identical original Cython decision models. It imports the exact
BayesFlow 2.1.0, 17-feature trial representation, dataset-max value
normalization, original parameter priors and classifier architecture from
run/fit_nn/tavares2017_bf. It does NOT adopt final_preprocessing's 50% QC.

**Research integrity:** These are new checkpoints. More simulations and
more stable validation can improve model recovery but cannot guarantee that
the original real-data DDM-Mul/RACE-Mul ranking will recur. Evaluation
includes S5 before making substantive claims from S6.

### Frozen empirical input
data/tavares2017/trial_eye.csv, SHA256
ff1479b10957aca922ec0792d87cb5bda0f6b19374d27c1eb6edc87c40bbf5be

25 subjects, 31,854 prepared trials, 31,853 retained by the old fixed
17-feature rule (one over-long RT), maximum trials in any subject 1,342.
Only option-fixation active-time enters the binary-eye models. Original
non-option fixation time does not enter the likelihood. No new exclusion.

### Frozen eight models and G2 priors

DDM: aDDM_1, aDDM_2, aDDM_t (Mul), aDDM_g (Add)
RACE: aRACE_1, aRACE_2, aRACE_t (Mul), aRACE_g (Add)
Same G2 priors as the old baseline: d U(1,50), a U(.5,5),
theta U(0,1), gamma U(0,2). Same BayesFlow SetTransformer(32),
cross-entropy classification, Adam lr 5e-4 clipnorm 1.5.
No changes to old simulator, Cython equations, feature adapter or priors.

### Pre-registered computational defaults for one formal run

- 128 online simulation batches per epoch x batch size 8 = 1,024
  synthetic participants per epoch, previously 30 x 8 = 240.
- Validation 512 synthetic participants, previously 64.
- 5,000 maximum epochs, ReduceLROnPlateau patience 8, EarlyStopping
  patience 30. Early stopping decides actual epochs; 5,000 is only a cap.
- 8 multiprocessing simulation workers, 9 requested CPUs.
- S5 model recovery: 2,000 independently simulated subjects **per
  generating model** (DDM4: 8,000; RACE4: 8,000; all8: 16,000).
- S6 uses the original 25 observed participants.
- Each model family gets one independent trained classifier.
- This is a single planned formal training budget, **not** an iterative
  hyperparameter search conditional on S6 model winners.

### One command on OSC login node

    cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
    git pull --ff-only
    cd run/fit_nn/tavares2017_legacy_formal
    bash submit.sh three

This submits 3 GPU S4 jobs + dependent S5/S6 jobs. Do not resubmit
the same family. Run bash submit.sh ddm (or race/all) for just one family.
All jobs use --account=PDE0114, --gres=gpu:1; request 9 CPUs/64G/100h
for S4 and 2 CPUs/32G/48h for S5/S6. These are Slurm limits, not runtime
predictions. Override ACCOUNT/CPUS/SIM_WORKERS/MEM/TIME/RECOVERY etc.
as needed.

For three separate Python commands inside a GPU allocation:

    python -u s4.py --family ddm
    python -u s5.py --family ddm
    python -u s6.py --family ddm

Identical commands apply to race and all. For no-GPU verification:
    python -u s4.py --family ddm --dry-run

### Output (all historical checkpoints stay untouched)

    outputs/tavares2017_legacy_formal/v1/
      classifier/
        aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/
           best.keras final.keras metadata.json history.csv
           prior_spec.json simulation_stats.json parallel_simulation.json
           diagnostics/metrics.json
           diagnostics/confusion_counts.csv
           diagnostics/per_model_metrics.csv
           diagnostics/predictions.csv
           real_data/model_probabilities.csv
           real_data/pairwise_log_bayes_factors.csv
        aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/
           (same)
        formal/
           (same, all eight)
      slurm/
        s4_*.out s4_*.err s5_*.out s6_*.out

The old (historical) 94.35%/84.53% model recoveries and old real-data
Mul-winner counts describe the **previous checkpoints only**, not these
newly retrained classifiers. No claim of model recovery, convergence or
real-data winners is made before S4/S5/S6 complete.
