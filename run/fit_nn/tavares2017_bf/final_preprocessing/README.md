# Tavares2017: one canonical final preprocessing dataset

Scope: data correctness ONLY; do not retrain or modify the eight models.
This step creates one isolated dataset in data/tavares2017_final, without
touching the previous baseline or six exploratory V2 profiles.

## Fixed scientific choices

To preserve the IDENTICAL binary-gaze Cython DDM/RACE models used for
Food Equal and Krajbich2010, the modeled reaction time is the SUM of
left/right option-fixation durations. It is NOT original button RT.
Original fixation codes 0 and 3 are off-item intervals and MUST NEVER
be mapped to model code 0 (which means RIGHT option).

- Source is pinned original Tavares Exp.1 expdata.csv and fixations.csv.
- Values use Thomas 2019 transformation 10 - 0.6 * abs(angle) and stay
  in original 1,4,7,10 units in this CSV; the existing G2 neural adapter
  divides by dataset max=10. v0=right, v1=left.
- Response: original -1=left,+1=right; modeled 1=left,0=right.
- Eye: original 1=left,2=right -> modeled 1=left,0=right.
  Off-item 0/3 durations are removed from the modeled active clock;
  adjacent same-option runs are merged.
- Original study QC: exclude trials where non-item duration is STRICTLY
  greater than 50% of full original RT.
- The unchanged legacy BF feature constraints exclude >=128 fixation
  runs or ceil(active_ms/3) >= floor(28287/3).
- Full raw RT, active RT, off-item RT, source values and trial IDs are
  retained as metadata and independently audited.

## Expected counts, from pinned original files

| Stage | Count |
|---|---:|
| Raw trials | 31,854 |
| >50% off-item QC exclusions | 1,691 |
| Existing model feature RT exclusion | 1 |
| FINAL valid S4/S6 trials | **30,162** |
| Participants | 25 |

The final file is independently passed through the exact existing
_shared.data.Experiment.real() adapter; any further trial exclusion
causes an error. This protects the final training/real-data pairing.

## OSC commands (CPU only)

From the git repository root:

~~~bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
git pull --ff-only
python -u run/fit_nn/tavares2017_bf/final_preprocessing/prepare_final.py
~~~

If the network can access the independent Thomas 2019 processed source,
verify ALL 31,854 original trials against its choice, option values,
raw RT and left/right gaze ratios:

~~~bash
python -u run/fit_nn/tavares2017_bf/final_preprocessing/prepare_final.py --fetch-thomas-reference
~~~

For offline work, supply local files:

~~~bash
python -u run/fit_nn/tavares2017_bf/final_preprocessing/prepare_final.py \
  --source-dir data/tavares2017/original \
  --thomas-reference /path/to/tavares2017_prep_all.csv
~~~

The GitHub CI workflow performs the complete source and Thomas validation
automatically, along with fixture tests.

## Outputs

~~~text
data/tavares2017_final/
  trial_eye.csv
  prep_manifest.json
  trial_audit.csv
  subject_audit.csv
~~~

Inspect prep_manifest.json: expected audited.final_trials=30162,
bayesflow_feature_check.retained_trials=30162 and,
if Thomas comparison was run,
original_reproduced_to_thomas.mismatch_count=0.

The original baseline data/checkpoints are UNCHANGED.

Do not yet start the existing S4 G2 wrapper with this final CSV, because
its output path is still shared with the old checkpoint. In Step 2 the
training launcher must choose a NEW unique checkpoint/output directory.
