#!/bin/bash
#SBATCH --account=PAS2943
#SBATCH --job-name=train
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

mkdir -p out

/users/PAS2197/benson31/yes/bin/python s4_train_range_huber.py