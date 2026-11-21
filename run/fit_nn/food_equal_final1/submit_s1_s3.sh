#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
mkdir -p "$ROOT/outputs/food_equal_final1/slurm"
MODELS=(aDDM_1 aDDM_2 aDDM_t aDDM_g aRACE_1 aRACE_2 aRACE_t aRACE_g)
command -v sbatch >/dev/null || { echo "Missing sbatch" >&2; exit 1; }
for model in "${MODELS[@]}"; do
  if [[ -e "$ROOT/outputs/food_equal_final1/$model/formal/best.keras" ]]; then
    echo "Existing checkpoint: $model. No jobs submitted; use standalone S2/S3." >&2
    exit 2
  fi
done
for model in "${MODELS[@]}"; do
  python -u s1.py --model "$model" --dry-run >/dev/null
done
for model in "${MODELS[@]}"; do
  job1=$(sbatch --parsable --account=PDE0114 --gres=gpu:1 --cpus-per-task=4 --time=100:00:00 --mem=64G --job-name="F1-S1-$model" --output="$ROOT/outputs/food_equal_final1/slurm/s1_$model_%j.out" --wrap="cd $HERE && python -u s1.py --model $model")
  job2=$(sbatch --parsable --account=PDE0114 --gres=gpu:1 --cpus-per-task=4 --time=08:00:00 --mem=32G --dependency=afterok:"$job1" --job-name="F1-S2-$model" --output="$ROOT/outputs/food_equal_final1/slurm/s2_$model_%j.out" --wrap="cd $HERE && python -u s2.py --model $model")
  job3=$(sbatch --parsable --account=PDE0114 --gres=gpu:1 --cpus-per-task=4 --time=04:00:00 --mem=32G --dependency=afterok:"$job2" --job-name="F1-S3-$model" --output="$ROOT/outputs/food_equal_final1/slurm/s3_$model_%j.out" --wrap="cd $HERE && python -u s3.py --model $model")
  echo "$model: S1=$job1 -> S2=$job2 -> S3=$job3"
done
