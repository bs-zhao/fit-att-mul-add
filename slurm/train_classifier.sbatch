#!/bin/bash
#SBATCH --account=PDE0114
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=att-class
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail
mkdir -p outputs/logs
python scripts/08_train_model_classifier.py
