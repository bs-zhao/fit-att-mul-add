#!/bin/bash
#SBATCH --account=PDE0114
#SBATCH --job-name=kraj-bf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-kraj-bf-%j.out
set -euo pipefail
# Submit from this folder. Select a Python 3.12/3.13 GPU PyTorch environment first.
export KERAS_BACKEND=torch
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
script=${1:-run_all.py}
if [ "$#" -gt 0 ]; then shift; fi
python -u "$script" "$@"
