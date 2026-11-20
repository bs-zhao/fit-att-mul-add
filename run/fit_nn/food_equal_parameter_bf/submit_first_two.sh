#!/usr/bin/env bash
# Two models only: aRACE_g (RACE-Add), aDDM_t (DDM-Mul).
# Each model: GPU S1 -> dependent GPU S2, S3, no racing.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
ACCOUNT="${ACCOUNT:-PDE0114}"
TIME="${TIME:-100:00:00}"
EVAL_TIME="${EVAL_TIME:-48:00:00}"
MEM="${MEM:-64G}"
EVAL_MEM="${EVAL_MEM:-32G}"
MODELS=(aRACE_g aDDM_t)
if [[ "${1:-both}" != "both" ]]; then
 case "$1" in aRACE_g|aDDM_t) MODELS=("$1");; *) echo "Usage: bash submit_first_two.sh [both|aRACE_g|aDDM_t]"; exit 2;; esac
fi
for model in "${MODELS[@]}"; do
 python -u s1.py --model "$model" --dry-run
done
LOG="$ROOT/outputs/food_equal_parameter_bf/v1/slurm"
mkdir -p "$LOG"
for model in "${MODELS[@]}"; do
 s1="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 --time="$TIME" \
  --cpus-per-task=4 --mem="$MEM" --job-name="Food-S1-$model" \
  --output="$LOG/s1_${model}_%j.out" --error="$LOG/s1_${model}_%j.err" \
  --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch MPLBACKEND=Agg && python -u s1.py --model $model")"
 s1="${s1%%;*}"
 s2="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 --time="$EVAL_TIME" \
  --cpus-per-task=2 --mem="$EVAL_MEM" --dependency="afterok:$s1" \
  --job-name="Food-S2-$model" \
  --output="$LOG/s2_${model}_%j.out" --error="$LOG/s2_${model}_%j.err" \
  --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch MPLBACKEND=Agg && python -u s2.py --model $model")"
 s2="${s2%%;*}"
 s3="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 --time="$EVAL_TIME" \
  --cpus-per-task=2 --mem="$EVAL_MEM" --dependency="afterok:$s2" \
  --job-name="Food-S3-$model" \
  --output="$LOG/s3_${model}_%j.out" --error="$LOG/s3_${model}_%j.err" \
  --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch MPLBACKEND=Agg && python -u s3.py --model $model --ppc-draws 0")"
 s3="${s3%%;*}"
 echo "[JOBS] $model: S1=$s1 S2=$s2 S3=$s3 (sequential)"
done
echo "[OUTPUT ROOT] $ROOT/outputs/food_equal_parameter_bf/v1"
echo "[LOGS] $LOG"
