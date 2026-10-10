# Krajbich 2010: minimal adaptation of `food_equal`

**Source of truth:** `run/fit_nn/food_equal/`. Every matching Python and
Slurm script here is copied from the source with ONLY the substitutions below.
No new CLI arguments, caching, data-processing algorithms, neural-network
architectures, or training hyperparameters have been introduced.

## Allowed deviations

1. Input CSV: `../../../data/krajbich2010/trial_eye.csv` in the main
   pipeline and `../../../../data/krajbich2010/trial_eye.csv` in
   `compare_regen/`. Original experiment RT is retained in this CSV as
   `rt_original_ms`; fitting uses **Scheme A**, the sum of fixation durations
   (exactly the source loader's definition).
2. All output paths: replace `outputs/food_equal` with
   `outputs/krajbich2010`; keep the same subdirectory and file naming.
3. `tools/model_info.py`: shared DDM `d=(0.06,2.8)`, `a=(2.3,10.5)`;
   shared ACC `d=(0.035,2.3)`, `a=(2.0,14.5)`. All theta, gamma,
   other fixed/free parameters, and model equations unchanged.
4. `tools/maxT.py`: `MAX_LEN_RT=28287` ms. `s1_gen1w.py` and
   `compare_regen/s1_regen.py`: `max_rt=28.287` s.
   `s2_getmaxRT.py`: `real_max_tp=28287`.
   The maximum is 20% above the largest **original recorded** RT.
5. Necessary **eight-model experiment selection**: remove `[1:2]` after
   the model list in `s4_train_range_huber.py`. In the four classifier
   feature/decode scripts, remove the SECOND assignment that restricted
   `model_names` to four DDMs. All eight models thus run; otherwise those
   scripts retain precisely the source implementation. In particular,
   the classifier's `pnames = model_infos[model_names[-1]]['free_pnames']`
   remains unchanged.
6. `tools/see_sample.py`: point its example pickle path to the
   corresponding Krajbich output.

The classifier continues to use **`data2param`** (the legacy classifier)
and the parameter decoder continues to use **`data2param_flow`**.
The 19 source-matched scripts have been checked for exact textual equality
after the above allowed transformations.

## Directory conventions

```text
data/krajbich2010/
  original/data_nature2010.dta
  trial_eye.csv              # transformed experimental input
outputs/krajbich2010/
  tests/                    # exploratory validation, separate from training
  s1_gen1/                  # training simulations
  s1_gen1_test/             # model recovery simulations
  s3_fe1/
  s3_fe1_maxT/
  dpsRH1_dp0.15/
  compare_regen/
```

The preprocessing / prior-calibration scripts remain in
`run/test_krajbich2010/`. Run its `s0_prepare_data.py` from the repository
root if the trial CSV must be regenerated.

## Running: EXACT same working-directory convention as food_equal

**Important:** The original scripts use relative paths and do NOT change
the process working directory. Therefore **do not invoke them from the
repository root**; switch to their containing directory first. The earlier
`--models`, `--n-round`, `--n-gen`, `--epochs` and
`--for-test` CLI options were removed to preserve source parity. Do not
reuse Slurm jobs that pass those arguments.

```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/run/fit_nn/krajbich2010
python s5_fe_real_data_maxT.py
python s1_gen1w.py
python s2_getmaxRT.py
python s3_fe1.py
sbatch train.sbatch
```

These are **successive stages**, not commands to start concurrently.
Wait for all eight models' 50 simulation files before feature extraction,
and wait for feature extraction to finish before training. The source
`train.sbatch` and its `20000` training epochs, early-stopping setting,
environment and Slurm directives are unchanged. This clone runs all eight
parameter networks sequentially within one invocation, instead of the
source's `[1:2]` single-network restriction.

After all eight parameter networks finish:

```bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/run/fit_nn/krajbich2010/compare_regen
python s1_regen.py
python s2_getmaxRT.py
python s3_fe1_maxT.py
sbatch train.sbatch
python s6_decode_maxT.py
```

Again wait for regeneration, feature extraction and classifier training
between successive stages. To use the original model recovery (MR)
path, manually change the scripts' original `for_test` variables to
`1` as in the source; no new CLI flags exist.

**If old Slurm array jobs are still pending/running, finish or cancel them
BEFORE `git pull`.** The prior added array wrappers passed CLI options
that the source-matched scripts no longer support. No existing
`outputs/` files have been deleted, moved, or overwritten by these GitHub
source changes.
