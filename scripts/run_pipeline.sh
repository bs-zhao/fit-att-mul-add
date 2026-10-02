#!/usr/bin/env bash
set -euo pipefail

python scripts/00_check_data.py
python scripts/01_simulate_training.py --model all
python scripts/02_build_training_features.py --model all
python scripts/03_train_parameter_decoder.py --model all
python scripts/04_build_real_features.py
python scripts/05_infer_parameters.py --model all

# Classifier-training resimulations.
python scripts/06_resimulate_posterior.py --model all --gen-start 1 --gen-end 20 --output-root outputs/resim_raw
python scripts/07_build_classifier_features.py --model all --input-root outputs/resim_raw --output-root outputs/classifier_features
python scripts/08_train_model_classifier.py

# Completely independent model-recovery set; never seen during classifier training.
python scripts/06_resimulate_posterior.py --model all --gen-start 101 --gen-end 105 --output-root outputs/recovery_raw
python scripts/07_build_classifier_features.py --model all --input-root outputs/recovery_raw --output-root outputs/recovery_features
python scripts/10_model_recovery.py --feature-root outputs/recovery_features

python scripts/09_decode_real_data.py
python scripts/11_summarize_comparison.py
