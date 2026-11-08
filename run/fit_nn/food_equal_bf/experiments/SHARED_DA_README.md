# Food Equal BF — shared d/a prior experiment (independent)

This experiment keeps the original 8 model equations, empirical input design,
all fixed parameters, BayesFlow architectures and training settings, but
**gives all eight models exactly the same d and a prior ranges**:

| Models | d | a | theta | gamma |
|---|---|---|---|---|
| DDM-NoAtt, RACE-NoAtt | U(0.1,5) | U(0.5,5) | fixed 1 | fixed 0 |
| DDM-AttOnly, RACE-AttOnly | U(0.1,5) | U(0.5,5) | fixed 0 | fixed 0 |
| DDM-Mul, RACE-Mul | U(0.1,5) | U(0.5,5) | U(0,1) | fixed 0 |
| DDM-Add, RACE-Add | U(0.1,5) | U(0.5,5) | fixed 1 | U(0,5) |

Source of unmodified model definitions: \`../food_equal/tools/model_info.py\`.
Default food_equal_bf S1-S6 remains unchanged. No files in either Krajbich
directory, old food_equal, or old pretrained outputs are modified.

### Entry point

\`python -u experiments/shared_da_classifier.py --stage STAGE --family FAMILY\`

Accepted S4/S5/S6 families are \`race\`, \`ddm\`, \`all\`. The entry point
temporarily overrides d/a on the independent in-process BF model table,
then reuses the **same existing S4, S5, S6** implementations and checkpoints.

All results are isolated under:

\`\`\`text
outputs/food_equal_bf/prior_sensitivity/
└── shared_d0p1to5_a0p5to5_theta0to1_gamma0to5/
    ├── classifier/
    │   ├── aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/
    │   │   ├── best.keras
    │   │   ├── prior_spec.json
    │   │   ├── metadata.json
    │   │   ├── diagnostics/        (S5)
    │   │   └── real_data/          (S6)
    │   ├── aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/
    │   └── formal/                 (all eight)
    └── cross_architecture/generated_aRACE_g/formal/seed_20261102/
\`\`\`

Never pass these checkpoints to ordinary \`s5_model_recovery.py\` or
\`s6_classify_real_data.py\` entry points without first applying the shared
prior override: the prior metadata check will correctly reject such usage.
Use \`experiments/shared_da_classifier.py\` for every stage.

### Run on OSC from \`run/fit_nn/food_equal_bf\`

Training: three independent classifiers, each a separate Slurm job:

\`\`\`bash
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=FE-SDA-DDM-S4 --output=FE-SDA-DDM-S4-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s4 --family ddm --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=FE-SDA-RACE-S4 --output=FE-SDA-RACE-S4-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s4 --family race --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=32G --job-name=FE-SDA-ALL-S4 --output=FE-SDA-ALL-S4-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s4 --family all --mode formal"
\`\`\`

Only after the **corresponding S4** has finished:

\`\`\`bash
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=30:00:00 --mem=32G --job-name=FE-SDA-DDM-S5 --output=FE-SDA-DDM-S5-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s5 --family ddm --mode formal --n-per-model 1000"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=30:00:00 --mem=32G --job-name=FE-SDA-RACE-S5 --output=FE-SDA-RACE-S5-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s5 --family race --mode formal --n-per-model 1000"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=50:00:00 --mem=32G --job-name=FE-SDA-ALL-S5 --output=FE-SDA-ALL-S5-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s5 --family all --mode formal --n-per-model 1000"
\`\`\`

Apply the same frozen classifiers to empirical food_equal participants:

\`\`\`bash
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=10:00:00 --mem=32G --job-name=FE-SDA-DDM-S6 --output=FE-SDA-DDM-S6-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s6 --family ddm --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=10:00:00 --mem=32G --job-name=FE-SDA-RACE-S6 --output=FE-SDA-RACE-S6-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s6 --family race --mode formal"
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=10:00:00 --mem=32G --job-name=FE-SDA-ALL-S6 --output=FE-SDA-ALL-S6-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage s6 --family all --mode formal"
\`\`\`

Test RACE-Add-generated data with the *same already trained* 4-DDM and 4-RACE
classifiers, including subjectwise Mul/Add Bayes factors. Requires **both**
4-model S4 checkpoints (does not use the 8-model classifier):

\`\`\`bash
sbatch --account=PAS2943 --gres=gpu:1 --cpus-per-task=4 --time=30:00:00 --mem=32G --job-name=FE-SDA-CROSS --output=FE-SDA-CROSS-%j.out --wrap="export KERAS_BACKEND=torch; python -u experiments/shared_da_classifier.py --stage cross --generator aRACE_g --mode formal --seed 20261102 --n 1000 --batch-size 25"
\`\`\`

For report plots on the separate four-DDM and four-RACE S5/S6 runs, after all
four are complete: \`python -u experiments/shared_da_classifier.py --stage report --family both --mode formal\`.

### Provenance/safety checks

- Every classifier has a \`prior_spec.json\` with an experiment ID, d/a/attention
  intervals, per-model free-prior dictionary, and model order.
- \`metadata.json\` records the same per-model free priors in \`prior_spec\`.
- S5/S6 and cross-architecture inference reject missing/incompatible metadata.
- Rerunning S4 without \`--overwrite\` refuses to replace an existing
  \`best.keras\`; \`--overwrite\` only affects this new experiment's checkpoint.
- Previous results under \`outputs/food_equal_bf/classifier/\` and the prior
  \`gamma_0_5_theta_0_1\` sensitivity subdirectory remain untouched.
- Training and full runtime validation must be run on the OSC host; this
  repository commit does **not** itself train neural networks.
