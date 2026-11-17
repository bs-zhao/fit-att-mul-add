# Tavares2017 — finalized-data G2 neural model comparison

This is ONLY the planned second step of the previously agreed minimal
workflow: keep the **same 8 models, same Cython likelihood, same BF model
architecture and same prior**, and make a single attempt at better S5
model recovery by increasing S4 training effort.

DO NOT use the older exploratory v2 full-timeline models. The ONLY input is
the independently audited 30,162-trial dataset:

    data/tavares2017_final/trial_eye.csv

Original Food Equal, Krajbich2010 and Tavares2017 checkpoints remain intact.
No parameter fitting (S1/S2), OOD project, or six-way sensitivity sweep.

## What changes relative to earlier Tavares BF training

| Setting | Old S4 | Final S4 |
|---|---:|---:|
| Eight models / equations / Cython | unchanged | unchanged |
| G2 priors | unchanged | unchanged |
| BayesFlow 2.1.0 / SetTransformer32 / CE / Adam(5e-4) | unchanged | unchanged |
| S4 batch size | 8 | 8 |
| S4 batches/epoch | 30 | 128 |
| Synthetic subjects/epoch | 240 | **1,024** |
| Validation subjects | 64 | **512** |
| ReduceLR patience | 4 | 6 |
| EarlyStopping patience | 10 | 20 |
| S5 recovery samples per generating model | configurable | default 500 |
| Data | old no-QC active clock | **final 50%-QC active clock** |

EarlyStopping/LR patience opt-in is implemented through two environment
variables read by the unchanged core common trainer. Historical training
without them remains 10/4.

The final training wrapper **imports** the original
experiments/normalized_value_prior_sweep.py and the independent parallel
trainer. It does NOT duplicate or rewrite the models, network or simulation
code. Original prior bounds d=(1,50), a=(.5,5), theta=(0,1),
gamma=(0,2) are enforced. Model priors 1/K; fixed legacy 17 BF features.

## Recommended OSC invocation: automatically submit 3 S4 + 3 S5 + 3 S6

~~~bash
cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
git pull --ff-only
cd run/fit_nn/tavares2017_bf
bash final_training/submit.sh three
~~~

The submitter checks the final CSV SHA256, 50%-QC, original source hashes,
all trial-level active clocks and the manifest BEFORE any sbatch request.
Then it submits one GPU S4 job per family, and schedules S5 and S6 after
that family's S4 succeeds. After all S5 and S6 jobs succeed, one CPU report
job runs automatically.

To run only one family:

~~~bash
bash final_training/submit.sh ddm
bash final_training/submit.sh race
bash final_training/submit.sh all
~~~

Do NOT submit a family twice while jobs are queued/running. Do NOT run
three plus the individual family submission commands simultaneously.

Small GPU allocation / shorter validation overrides:

~~~bash
BATCHES=96 VALIDATION=256 TRAIN_BATCH=4 \
  bash final_training/submit.sh three
~~~

The defaults allocate one GPU, 9 CPUs and 64GB for S4, 2 CPUs/32GB and
one GPU for each S5/S6. If your OSC limits differ, override
ACCOUNT, CPUS, SIM_WORKERS, TRAIN_MEM, EVAL_MEM, TRAIN_TIME, EVAL_TIME.
The maximum formal epochs is 5,000; EarlyStopping usually ends earlier.

For S4-only submission, and later manually execute S5/S6:

~~~bash
FOLLOWUPS=0 bash final_training/submit.sh three
python -u final_training/run.py --stage s5 --family ddm --n-per-model 500
python -u final_training/run.py --stage s6 --family ddm
~~~

The S5 and S6 commands above need a GPU allocation; use sbatch if not
already on a GPU compute node.

For **non-training** checks:

~~~bash
python -u final_training/run.py --stage s4 --family ddm --dry-run
python -m unittest discover -s final_training/tests -v
bash -n final_training/submit.sh
~~~

Small isolated training smoke-test (GPU allocation):

~~~bash
python -u final_training/run.py --stage s4 --family ddm --mode test \
  --epochs 2 --batches 2 --batch-size 2 --validation 8 --sim-workers 2
~~~

## Output directory (NOT the historical ones)

All outputs are under:

    /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/outputs/
      tavares2017_bf_final/g2_train_v1/

Specifically:

    outputs/tavares2017_bf_final/g2_train_v1/
      classifier/
        aDDM_1__aDDM_2__aDDM_t__aDDM_g/formal/
          best.keras             # S4 best
          final.keras            # S4 final
          history.csv
          metadata.json
          prior_spec.json
          simulation_stats.json
          parallel_simulation.json
          diagnostics/           # S5
            metrics.json
            confusion_counts.csv
            confusion_normalized.csv
            per_model_metrics.csv
            predictions.csv
            training_loss.png
            ...
          real_data/             # S6
            model_probabilities.csv
            pairwise_log_bayes_factors.csv
            trial_audit.csv
            model_probabilities.png
        aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal/
          (same S4/S5/S6 structure)
        formal/                  # combined 8 models
          (same S4/S5/S6 structure)
      summary/
        model_recovery_comparison.csv
        per_model_recovery_comparison.csv
        report.json
      slurm/
        S4_ddm_JOBID.out    S4_ddm_JOBID.err
        S4_race_JOBID.out   S4_race_JOBID.err
        S4_all_JOBID.out    S4_all_JOBID.err
        S5_*_JOBID.out      S6_*_JOBID.out
        report_JOBID.out

Run a read-only comparison any time (no GPU):

~~~bash
cd run/fit_nn/tavares2017_bf
python -u final_training/report.py
~~~

The report compares the new S5 model recovery accuracy and per-model recall
with the previous published Tavares BF baseline outputs if available.
It records any difference in held-out sample counts rather than implying
identical uncertainty.

## Interpretation

Only one S4 training improvement attempt is planned. If the RACE and joint
model recovery materially improves, use the new model comparison. If
recovery remains limited (especially aRACE_t), retain the honest results
rather than launching unrelated OOD or complex sensitivity projects.

Do not confuse improved neural recovery with proof that every real-data
posterior/Bayes factor is calibrated. The final analysis should disclose
that the modeled decision clock is option-fixation time, not full raw RT.
