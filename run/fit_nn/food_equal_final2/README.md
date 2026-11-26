# food_equal_final2 (BayesFlow 2.1.0) — independent S4–S6

Final2 is exactly Final1 with **only the free gamma prior**
`gamma ~ Uniform(0,2)` (instead of `Uniform(0,10)`) for both DDM-Add and RACE-Add.
Other priors are unchanged: d=U(0.1,5), a=U(0.5,10), theta=U(0,1).
Identical shared fixed parameters, datasets, input normalization, simulator, model
definitions, train hyperparameters, held-out S5 budgets, S6 decoding, and
the same seed defaults. S4 classifiers MUST be retrained since the generating
model prior has changed. No Final1 checkpoint is read or modified.

Outputs are isolated in `outputs/food_equal_final2/`.
Training generates three independent S4 classifiers (DDM4, RACE4, ALL8).
S5 uses the matching Final2 S4 checkpoints for model recovery, and S6
compares real data with those checkpoints.

### Slurm: one command for all S4 → S5 → S6 jobs

```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/run/fit_nn/food_equal_final2
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
