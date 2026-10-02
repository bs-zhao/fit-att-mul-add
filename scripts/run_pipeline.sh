#!/usr/bin/env bash
set -euo pipefail

python scripts/00_check_data.py
python scripts/01_simulate_training.py --model all
python scripts/02_build_training_features.py --model all
python scripts/03_train_parameter_decoder.py --model all
python scripts/04_build_real_features.py
python scripts/05_infer_parameters.py --model all
python scripts/06_resimulate_posterior.py --model all
python scripts/07_build_classifier_features.py --model all
python scripts/08_train_model_classifier.py
python scripts/10_model_recovery.py
python scripts/09_decode_real_data.py
python scripts/11_summarize_comparison.py
