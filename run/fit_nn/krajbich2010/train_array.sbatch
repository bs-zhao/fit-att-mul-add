#!/bin/bash
#SBATCH --account=PAS2943
#SBATCH --job-name=kraj-fit
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-7

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODELS=(aDDM_1 aDDM_2 aDDM_t aDDM_g aRACE_1 aRACE_2 aRACE_t aRACE_g)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
/users/PAS2197/benson31/yes/bin/python s4_train_range_huber.py --models "$MODEL"
