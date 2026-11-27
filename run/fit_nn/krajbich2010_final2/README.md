# krajbich2010_final2 (BayesFlow 2.1.0) — independent S4–S6

Final2 is exactly Final1 with **only the free gamma prior**
`gamma ~ Uniform(0,2)` (instead of `Uniform(0,10)`) for both DDM-Add and RACE-Add.
Other priors are unchanged: d=U(0.1,5), a=U(0.5,10), theta=U(0,1).
Identical shared fixed parameters, datasets, input normalization, simulator, model
definitions, train hyperparameters, held-out S5 budgets, S6 decoding, and
the same seed defaults. S4 classifiers MUST be retrained since the generating
model prior has changed. No Final1 checkpoint is read or modified.

Outputs are isolated in `outputs/krajbich2010_final2/`.
Training generates three independent S4 classifiers (DDM4, RACE4, ALL8).
S5 uses the matching Final2 S4 checkpoints for model recovery, and S6
compares real data with those checkpoints.

### Slurm: one command for all S4 → S5 → S6 jobs

```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/run/fit_nn/krajbich2010_final2
bash submit_s4_s6.sh
```

This submits 3 S4 jobs, each with an S5 job dependent on its successful
completion, followed by S6 dependent on S5. 9 jobs per dataset, with Slurm
`afterok` dependencies. Default S4 time=100h, S5=48h, S6=8h.
Only submit once. No scripts overwrite existing checkpoints by default.

### Slurm: manually submit each stage separately

```bash
bash submit.sh s4 three
# After all three S4 jobs have completed successfully:
TIME=48:00:00 MEM=32G bash submit.sh s5 three
# After S5 diagnostics checked:
TIME=08:00:00 MEM=32G bash submit.sh s6 three
```

For quick preflight: `python -u s4.py --family ddm --dry-run`.


### Cross-architecture simulations (Final2 ONLY)

Both scripts use **the existing Final2 S4 DDM4 checkpoint** and Final2 `gamma ~ U(0,2)`, with all other settings identical to Final1.

**Prior-generated RACE-Add → DDM4** (requires only Final2 S4 DDM4):
```bash
python -u cross_race_add_vs_ddm4.py --n 1000 --batch-size 25
```
The RACE-Add generator draws d~U(0.1,5), a~U(0.5,10), gamma~U(0,2).
The output is isolated under `outputs/krajbich2010_final2/cross_architecture/generated_aRACE_g__tested_DDM4/formal/`.
Use `--dry-run` to inspect inputs and the intended checkpoint.

**S3-fitted RACE-Add → DDM4** (requires **Final2** RACE-Add S1 and S3 outputs, not Final1):
```bash
# If Final2 S1 has never been submitted:
bash submit_s1_s3_race_add.sh
# After successful S3:
python -u cross_s3_fitted_race_add_vs_ddm4.py --repeats 10 --batch-size 10
```
Uses each subject's **Final2 S3 joint posterior mean** (d,a,gamma), fixed for 10 independent behavior simulations, preserving the subject's empirical value and gaze schedules. Requires `outputs/krajbich2010_final2/aRACE_g/formal/real_data/posterior_samples.npz`. It validates that `gamma <= 2` and checks checkpoint metadata; no Final1 posterior or checkpoint is used. Output is isolated under `outputs/krajbich2010_final2/cross_architecture/S3_fitted_aRACE_g__tested_DDM4/formal/`.

If the S1 checkpoint already exists, do not rerun `submit_s1_s3_race_add.sh`; submit S2 and S3 individually:
```bash
bash submit.sh s2 aRACE_g
# After S2 completes:
bash submit.sh s3 aRACE_g
```
Both experiments retain per-simulation probabilities, log BF(Mul/Add), summary statistics and a histogram; S3-fitted additionally creates per-subject summaries. Do not submit the same experiment twice with the same seed/repeats unless you have archived the first output.
