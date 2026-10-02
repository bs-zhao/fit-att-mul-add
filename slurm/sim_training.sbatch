#!/bin/bash
#SBATCH --account=PDE0114
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=att-sim
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail
MODEL=${1:?model required}
START=${2:-1}
END=${3:-10}
mkdir -p outputs/logs
python scripts/01_simulate_training.py --model "$MODEL" --gen-start "$START" --gen-end "$END"
