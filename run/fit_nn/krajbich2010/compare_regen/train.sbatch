#!/bin/bash
#SBATCH --account=PAS2943
#SBATCH --job-name=train
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p out

/users/PAS2197/benson31/yes/bin/python s4_train_classifier_maxT.py