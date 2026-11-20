# krajbich2010_final1 — S1-S6 (Unified Final1)

This is the isolated krajbich2010 runner for the **same eight mathematical
models and priors** across food_equal, krajbich2010 and tavares2017.

Unified model priors (continuous uniform): `d=(0.1,5)`,
`a=(0.5,10)`, `theta=(0,1)`, `gamma=(0,10)`.
All fixed model parameter values and multiplicative vs additive variants
come from the SAME base catalogue at `run/fit_nn/krajbich2010/tools/model_info.py`,
with shared `ndt=0`, `s=1`, `lam=-1`, and RACE `b=0`.
No new non-decision-time free parameter was added.

**Value normalization**: after the existing dataset-specific gaze/row
preprocessing, calculate one global maximum of v0/v1 across all retained
rows; for EACH stimulus value, `v_normalized=v_original/global_max`.
This is simple division, *not* min-max scaling and not subject-wise.
Same normalized values are sent into the simulator and real-data network
features, with feature scaling for v0/v1 both set to 1; raw CSV untouched.

Each stage reuses the pre-existing, versioned BayesFlow 2.1.0 code at
`run/fit_nn/krajbich2010_bf/s1...s6`, with process-local Final1 configuration
installed BEFORE importing the stage. Old experiments and checkpoints are
not modified or loaded.

- S1: 8 individual model parameter posterior networks (2/3 dimensions).
- S2: fresh held-out parameter recovery & SBC.
- S3: real-subject posterior decode; optional PPC.
- S4: model comparison networks for 4-DDM, 4-RACE, and all 8.
- S5: balanced held-out model recovery & calibration.
- S6: empirical model comparison & pairwise log Bayes factors.

### Run on OSC
```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
git pull --ff-only
cd run/fit_nn/krajbich2010_final1

# First two parameter models: RACE Add and DDM Mul
bash submit.sh s1 two
# AFTER S1 completed and checkpoints verified:
bash submit.sh s2 two
# AFTER S2 diagnostics reviewed:
bash submit.sh s3 two

# Train three independent classifiers
bash submit.sh s4 three
# AFTER S4 completed:
bash submit.sh s5 three
# AFTER S5 reviewed:
bash submit.sh s6 three

# Scale parameter fitting to all eight:
bash submit.sh s1 all
bash submit.sh s2 all
bash submit.sh s3 all
```

Each submission runs only the requested stage, with NO automatic dependent
jobs. Check `sacct -j <jobids> --format=JobID,State,ExitCode -X` before the next
stage. Submit for one dataset or any model/family independently.

Direct command example:
```bash
python -u s1.py --model aRACE_g
python -u s2.py --model aRACE_g
python -u s3.py --model aRACE_g --ppc-draws 0
python -u s4.py --family ddm
python -u s5.py --family ddm
python -u s6.py --family ddm
```

S1 defaults: 5000 max epochs, 50 batches × 64 participants,
validation 2000. S2 defaults: 2000 held-out participants, 1000 draws.
S3 defaults: 2000 draws, PPC off (enable with --ppc-draws).
S4: 50 × 128 participants per epoch and validation 2000.
S5: 2000 held-out participants per generating model.

Output: `outputs/krajbich2010_final1/`:
```text
<model>/formal/{best.keras,metadata.json,final1_manifest.json,history.csv,
  diagnostics/,real_data/}
classifier/formal/                 # all 8
classifier/aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/
classifier/aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/
slurm/
```

Check `metadata.json.prior_spec`: profile=`unified_final1_v1`,
raw value max, all free/fixed priors, model definitions and dataset SHA.
Training convergence is diagnosed empirically and is not guaranteed by
a 5000 epoch upper limit; S2/S5 need to pass recovery checks.

**Tavares source**: `data/tavares2017/trial_eye.csv` (the original
Tavares experiment with its existing trial filters), NOT
`data/tavares2017_final/trial_eye.csv`. Choose the other source only by
explicitly passing --input to all stages, including S1/S4.
