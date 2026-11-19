# Final Tavares2017: 2x2 Cross-Inference (No Retraining)

Purpose: determine whether the loss of DDM-Mul support arises mainly
from QC/data, from the newly trained classifier, or from an interaction.

Only S6 with frozen old and new G2 classifiers. No S4, S5 or new model
simulation. Original models, checkpoint weights, priors and data untouched.

## Four combinations

| Checkpoint | Old data (no 50% QC) | New data (50% QC) |
|---|---|---|
| Old G2 classifier | Reuse historical S6 | Run cross-S6 |
| New G2 classifier | Run cross-S6 | Reuse final S6 |

Default covers DDM and RACE four-model classifiers. You can additionally
include the eight-model classifier with the three option.

The real saved checkpoint shapes are old max_trials=1342 and new=1339.
All data are passed with the original 17 features, summary mask and
n_trials. No trials are artificially removed or truncated to equalize
these shapes. If a SetTransformer cannot handle the other sequence
length, cross-inference stops with an explicit error.

The original source hashes, final data SHA256, model parameter priors,
value normalizers, 17 feature names and feature scales are all checked
before launching inference. The exact historical original data has one
long-duration trial removed in the S6 feature adapter, resulting in
31,853 usable trials. The new final input has 30,162. This expected
difference is explicitly checked.

## One-shot OSC submission

On the OSC login node, from your GitHub repository:

    cd /fs/scratch/PAS2943/Benson/Projects/EEG-EYE
    git pull --ff-only
    cd run/fit_nn/tavares2017_bf
    bash final_training/submit_cross.sh both

This submits 4 GPU inference jobs plus one dependent CPU summary job.
It does not train neural networks.

For all three classifier families (DDM, RACE, joint eight):
    bash final_training/submit_cross.sh three

Do not run both and three simultaneously; that would submit duplicate
DDM/RACE jobs.

## Individual run commands (GPU allocation)

    python -u final_training/cross_inference.py --family ddm --checkpoint old --data new
    python -u final_training/cross_inference.py --family ddm --checkpoint new --data old
    python -u final_training/cross_inference.py --family race --checkpoint old --data new
    python -u final_training/cross_inference.py --family race --checkpoint new --data old

For no-GPU checks, add --dry-run to each command.

After completion, CPU-only:
    python -u final_training/cross_report.py

Each S6 job requests 1 GPU, 2 CPUs, 32 GB memory and up to 8 hours.
These are allocation limits, not execution-time predictions.
Override ACCOUNT, CPUS, MEM, TIME or BATCH as needed.

## Output root

    /fs/scratch/PAS2943/Benson/Projects/EEG-EYE/outputs/tavares2017_bf_final/g2_train_v1/cross_inference/

Contains:

    cross_inference/
      cells/
        ddm/
          old_classifier__new_data/
            model_probabilities.csv
            real_data_trial_audit.csv
            metadata.json
          new_classifier__old_data/
            model_probabilities.csv
            real_data_trial_audit.csv
            metadata.json
        race/
          old_classifier__new_data/
          new_classifier__old_data/
        all/                            # optional
      summary/
        four_cells.csv
        per_subject.csv
        two_by_two_decomposition.csv
        ddm_vs_race_mul.csv
        provenance.json
      slurm/
        *.out
        *.err

First inspect summary/ddm_vs_race_mul.csv: for each of the four
cells it directly displays the number of Mul-winning subjects in DDM
and RACE, and DDM minus RACE.

The 2x2 decomposition defines A=old model+old data, B=old model+new
data, C=new model+old data, D=new model+new data.
Data contrast at old checkpoint: B-A; at new checkpoint: D-C.
Training contrast on old data: C-A; on new data: D-B.
Interaction: D-C-B+A. These are descriptive, not causal proof.

Historical diagonal S6 CSV files are READ-ONLY, as are best.keras,
metadata.json, input data and all existing classifier outputs.

If an old checkpoint or diagonal S6 is missing, this will terminate
with a missing-file error. Restore the original old baseline directory
before running; DO NOT retrain it.
