#!/bin/bash
#SBATCH --account=PDE0114
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=att-resim
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail
MODEL=${1:?model required}
START=${2:-1}
END=${3:-20}
mkdir -p outputs/logs
python scripts/06_resimulate_posterior.py --model "$MODEL" --gen-start "$START" --gen-end "$END"
