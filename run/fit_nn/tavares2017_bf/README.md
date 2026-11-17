# Tavares2017 Experiment 1: G2 matched-prior BayesFlow

Independent third-data-set implementation of the Food Equal / Krajbich2010
**eight identical c_bifood DDM/RACE models**, with exactly the same G2
dataset-max value normalizer, 17 trial features, SetTransformer/CouplingFlow,
BayesFlow 2.1.0 workflow, 1 ms Cython decision dynamics and priors.

**Reference:** Thomas et al. (2019), using Tavares et al. (2017) Experiment 1.
Raw data: https://github.com/glamlab/gaze-bias-differences/tree/master/data/tavares_2017_fns/original

## 0. Prepare original data

From this directory, on a login node with access to GitHub:

\`\`\`bash
python -u prepare_data.py
python -m unittest discover -s tests -v
\`\`\`

If direct download is blocked, copy the original \`expdata.csv\` and
\`fixations.csv\` locally, then use:

\`\`\`bash
python -u prepare_data.py --source-dir /path/to/original
\`\`\`

Produces \`<repo>/data/tavares2017/trial_eye.csv\` and \`prep_manifest.json\`,
with upstream SHA256 hashes, full-RT/active-RT accounting, and an audit.
The model input includes every subject and trial with valid option fixation;
no random trial subsampling is performed.

### Critical decision-clock distinction

The original raw fixation codes are 1=left, 2=right, 0=off-item, 3=initial
off-item. The legacy \`c_bifood\` implementation accepts only 1=left and
0=right; coding original off-item 0 as model 0 would **invent right gaze**.

To keep *exactly the same eight model equations*, this adapter removes the
non-option gaze intervals and concatenates the resulting item fixations.
Hence the modeled RT is **total item-fixation time** (active evidence time),
not the original stimulus-to-button RT. Both times remain recorded in the
prepared table and manifest. This is a documented change of the observed
decision-clock definition, NOT a change to DDM/RACE drift equations. It
must be disclosed in any article comparing this dataset with food datasets;
do not report this as a full-RT fit. Raw button RT can be recovered from
\`rt_raw_ms\`.

Item values are converted from angular deviation in degrees using
\`10 - 0.6*abs(angle)\`, so 0/5/10/15 degrees map to values 10/7/4/1.
The G2 pipeline **then** divides both options by the dataset-wide max 10.
Storage is \`v0=right, v1=left\`; choice \`1=left, 0=right\`. The upstream
source's original choice is \`-1=left, +1=right\`.

## 1. Candidate models and exactly matched G2 priors

\`\`\`text
ddm:  aDDM_1 (NoAtt), aDDM_2 (AttOnly), aDDM_t (Mul), aDDM_g (Add)
race: aRACE_1 (NoAtt), aRACE_2 (AttOnly), aRACE_t (Mul), aRACE_g (Add)
all:  all eight
\`\`\`

\`\`\`text
d     ~ Uniform(1,50)
a     ~ Uniform(0.5,5)
theta ~ Uniform(0,1)    [Mul]
gamma ~ Uniform(0,2)    [Add]
\`\`\`

Only dataset inputs differ. The **identical** c_bifood Cython simulators and
Krajbich2010 legacy model specification (fixed \`s=1\`, \`ndt=0\`,
\`lam=-1\`, \`b=0\` for RACE, same baseline values of theta/gamma) are imported
rather than redefining equations. S4 uses uniform model-class priors.

Training data match the number of retained trials per empirical subject.
Tavares has around 1,300 trials per person, so the default BF training
batch was reduced to 8 and validation batch to 64 for S4. On lower-memory
GPUs, try \`--batch-size 4 --validation 16\`. This does not alter the scientific
model or trial counts.

## 2. Train the two four-way classifiers and one eight-way classifier

Run these in the directory containing this README, after preparation:

\`\`\`bash
export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u experiments/parallel_normalized_value_prior_sweep.py --sim-workers 8 --stage s4 --family ddm  --mode formal --batch-size 8 --validation 64
python -u experiments/parallel_normalized_value_prior_sweep.py --sim-workers 8 --stage s4 --family race --mode formal --batch-size 8 --validation 64
python -u experiments/parallel_normalized_value_prior_sweep.py --sim-workers 8 --stage s4 --family all  --mode formal --batch-size 8 --validation 64
\`\`\`

These are independent jobs: submit one per GPU to Slurm, not in a sequence
on one allocated GPU. Example (replace account and working directory):

\`\`\`bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/run/fit_nn/tavares2017_bf
for family in ddm race all; do
    sbatch --account=PDE0114 --gres=gpu:1 --cpus-per-task=9 --time=100:00:00 --mem=32G \
      --job-name="Tavares-G2-\${family}" \
      --wrap="cd $PWD && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u experiments/parallel_normalized_value_prior_sweep.py --sim-workers 8 --stage s4 --family \${family} --mode formal --batch-size 8 --validation 64"
done
\`\`\`

The 8-worker simulator may be much slower on 1,300-trial subjects than on
Food Equal; monitor \`parallel_simulation.json\` and Slurm logs. Formal
training uses unchanged early stopping. Training results are independent
per family and use \`best.keras\`. Do not reuse old checkpoints.

## 3. After S4: S5 held-out model recovery

\`\`\`bash
for family in ddm race all; do
  python -u experiments/normalized_value_prior_sweep.py \
    --stage s5 --family "$family" --mode formal --n-per-model 1000 --batch-size 4
done
\`\`\`

S5 saves confusion matrices, accuracy, probability calibration, and related
diagnostics. Inspect calibration and prior predictive non-hit rates before
interpreting S6 as evidence for Mul/Add.

## 4. After S4: S6 classify the 25 real participants

\`\`\`bash
for family in ddm race all; do
  python -u experiments/normalized_value_prior_sweep.py \
    --stage s6 --family "$family" --mode formal --batch-size 4
done
\`\`\`

The S6 input is generated automatically from \`data/tavares2017/trial_eye.csv\`.
No additional per-subject file is needed. S6 writes:

- \`real_data/model_probabilities.csv\`: every subject's 4- or 8-model PMPs;
- \`real_data/pairwise_log_bayes_factors.csv\`: ordered subject-level log BFs;
- \`real_data/trial_audit.csv\`, \`model_prior.json\` and a heatmap.

All runs use an input SHA256 and normalized prior-spec compatibility checks,
so a classifier cannot silently be reused after regenerating altered data.

Results are under:

\`\`\`text
outputs/tavares2017_bf/prior_sensitivity/
  normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0/
    classifier/
      aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/
      aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/
      formal/              # eight-model classifier
\`\`\`

## 5. Optional cross architecture and individual parameter recovery

\`\`\`bash
python -u experiments/normalized_value_prior_sweep.py --stage cross --mode formal --generator aRACE_g --n 1000 --batch-size 4
python -u experiments/normalized_value_prior_sweep.py --stage report --family both --mode formal

python -u experiments/normalized_parameter_recovery.py --stage s1 --model aDDM_t --mode formal --batch-size 8
python -u experiments/normalized_parameter_recovery.py --stage s2 --model aDDM_t --mode formal --n-test 1000 --draws 1000 --batch-size 4
\`\`\`

Note that S1/S2 is *parameter* recovery; S5 is *model* recovery. The
models remain independent of Thomas et al.'s GLAM implementation.
