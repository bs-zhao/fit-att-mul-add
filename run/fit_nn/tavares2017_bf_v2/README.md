# Tavares2017 V2: model comparison consistency and data sensitivity

This study is intentionally **independent** of the completed baseline.

- Baseline: run/fit_nn/tavares2017_bf
- Baseline follow-up tests: run/fit_nn/tavares2017_bf/next_steps
- New V2: run/fit_nn/tavares2017_bf_v2
- All V2 input data: data/tavares2017_v2/PROFILE/
- All V2 results: outputs/tavares2017_bf_v2/PROFILE/

The **same eight models** are used, with priors d=U(1,50), a=U(0.5,5),
theta=U(0,1), gamma=U(0,2), BayesFlow 2.1.0, SetTransformer32,
cross entropy, Adam(5e-4), and 1 ms Cython. The V2 full-timeline
simulator adds an explicit neutral gaze state (-1), setting deterministic
DDM drift and both RACE drift rates to zero while diffusion remains active.
For all left/right eye states, the Cython equations are identical to baseline
and verified through same-seed parity tests.

## Stage A: audit existing S4 checkpoints, prior data, and 4-vs-8 consistency

On OSC:

~~~bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
git pull --ff-only
cd run/fit_nn/tavares2017_bf
bash prepare_next_steps.sh
~~~

The preparation script checks original source data and code/checkpoint
alignment, then prints a GPU sbatch command. For direct submission:

~~~bash
sbatch --account=PDE0114 --gres=gpu:1 --time=24:00:00 --mem=32G \
 --job-name=Tav-Consistency \
 --wrap="export KERAS_BACKEND=torch; python -u next_steps/consistency.py --n-per-model 32 --batch-size 4 --data both"
~~~

Use this before launching large follow-up training. It feeds EXACTLY the same
synthetic subject sets into all 3 previously trained classifiers and measures
within-family four-way versus eight-way conditional probability differences.
The same comparison is performed for all 25 real participants.

Outputs:

~~~text
outputs/tavares2017_bf/next_steps/alignment.json
outputs/tavares2017_bf/next_steps/data_audit/summary.json
outputs/tavares2017_bf/next_steps/data_audit/trial_level.csv
outputs/tavares2017_bf/next_steps/data_audit/subject_level.csv
outputs/tavares2017_bf/next_steps/consistency/summary.json
outputs/tavares2017_bf/next_steps/consistency/synthetic_paired_posteriors.csv
outputs/tavares2017_bf/next_steps/consistency/real_paired_posteriors.csv
~~~

If consistency is also poor on the paired synthetic datasets, neural
classifier evidence approximation requires additional attention even
without any real-data preprocessing concerns.

## Stage B: prepare six controlled data profiles and Cython extension

From the repository root:

~~~bash
python -u run/fit_nn/tavares2017_bf/prepare_data.py
cd run/fit_nn/tavares2017_bf_v2
python -u prepare_variants.py
python setup_cython.py build_ext --inplace
python -m unittest discover -s tests -p test_prepare_variants.py -v
~~~

The usual baseline c_bifood package must also be installed/importable.
GitHub Actions independently compiles both Cython packages and verifies
all eight binary-model stochastic trajectories match bitwise.

| Profile | Decision clock | Values | Drop >50% off-item trials? |
|---|---|---|---|
| active_thomas_all | Item fixation only | Thomas (1,4,7,10) | No |
| active_thomas_qc | Item fixation only | Thomas (1,4,7,10) | Yes |
| active_relative_qc | Item fixation only | Relative (0,1/3,2/3,1) | Yes |
| full_thomas_all | Entire original RT | Thomas (1,4,7,10) | No |
| full_thomas_qc | Entire original RT | Thomas (1,4,7,10) | Yes |
| full_relative_qc | Entire original RT | Relative (0,1/3,2/3,1) | Yes |

Each input profile gets its own CSV and SHA256 manifest. The full clock
preserves all original off-item 0/3 intervals as gaze=-1. The old simulator
is NEVER used to interpret -1 as right gaze. The choice code is still 1=left,
and option order is still [right,left]. The original data have 31,854 trials;
a 50% quality exclusion removes 1,691, retaining 30,163.

Across V2 profiles, the maximum RT cutoff and RT feature scale are
consistently set to 50 seconds so comparisons WITHIN V2 do not confound
this difference. Direct baseline-versus-V2 contrasts therefore require
attention to the V1 RT cutoff (28.287 seconds) and network feature scale.
This is an explicit sensitivity analysis, not an assertion that both clocks
represent equivalent likelihood models.

The zero-neutral-drift/noise-active construction is a hypothesis for off-item
processing; it is not presented as the unique interpretation of all neutral
states in the original task.

## Stage C: S4 training, S5 recovery, S6 real subjects

Run in run/fit_nn/tavares2017_bf_v2.

Submit one profile's three classifier training jobs and automatically queue
independent S5 + S6 jobs with Slurm afterok dependencies:

~~~bash
bash submit_variant.sh full_thomas_qc three
~~~

Alternatively submit each family once:

~~~bash
bash submit_variant.sh full_thomas_qc ddm
bash submit_variant.sh full_thomas_qc race
bash submit_variant.sh full_thomas_qc all
~~~

Do NOT submit both forms for the same profile: they share checkpoint paths.

Defaults: 8 simulation workers, S4 batch size 8, 128 batches/epoch (1,024
synthetic subjects per epoch), 512 validation subjects, 5000 epoch maximum,
unchanged early stopping and 512 held-out S5 subjects per generating model.

Override resource and training settings without editing code:

~~~bash
BATCHES=64 VALIDATION=256 TRAIN_BATCH=4 RECOVERY_N=256 \
 bash submit_variant.sh active_thomas_qc three
~~~

For only S4 jobs without automatically submitted S5/S6:

~~~bash
FOLLOWUPS=0 bash submit_variant.sh full_relative_qc three
~~~

Smoke-test a tiny S4 model on an allocated GPU without formal checkpoint
contamination:

~~~bash
python -u experiments/parallel_normalized_value_prior_sweep.py \
 --sim-workers 2 --stage s4 --family ddm --mode test \
 --input ../../../data/tavares2017_v2/full_thomas_qc/trial_eye.csv \
 --epochs 1 --batches 1 --batch-size 2 --validation 4
~~~

## Stage D: comparisons — recommended priority

1. Existing baseline: paired synthetic 4/8 posterior consistency (Stage A).
2. active_thomas_qc versus full_thomas_qc: decision clock, with same
   quality filtering, feature scale, and Thomas values.
3. full_thomas_qc versus full_relative_qc: value-zero sensitivity, with
   identical clock and quality filter.
4. active_thomas_all versus active_thomas_qc: off-item quality filter only.
5. full_thomas_all versus full_thomas_qc: off-item filter under full clock.
6. active_relative_qc: test whether value scale interacts with decision clock.

An excellent S5 model recovery rate alone does not guarantee accurate
approximate Bayes factors on out-of-distribution real behavior. Keep all
original checkpoints and do not interpret new V2 odds as directly
interchangeable with the V1 baseline.

## Output folder layout

~~~text
outputs/tavares2017_bf_v2/PROFILE/prior_sensitivity/
  normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0/
    classifier/
      aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/
      aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/
      formal/                  # combined 8-way classifier
~~~

For each classifier S4 writes best.keras, metadata.json, history.csv;
S5 writes diagnostics/; S6 writes real_data/model_probabilities.csv,
real_data/pairwise_log_bayes_factors.csv and trial_audit.csv.
