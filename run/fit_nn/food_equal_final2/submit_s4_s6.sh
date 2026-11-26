#!/usr/bin/env bash
# Submit the three Final2 S4 classifiers; schedule S5 and S6 after S4 finishes.
# Per family: S4 -> S5 -> S6. Never touches any Final1 outputs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DATASET="food_equal"
OUT="$ROOT/outputs/${DATASET}_final2/slurm"
mkdir -p "$OUT"
cd "$HERE"
for family in ddm race all; do
  python -u s4.py --family "$family" --dry-run >/dev/null
done
for family in ddm race all; do
  j4=$(sbatch --parsable --account="${ACCOUNT:-PDE0114}" --gres=gpu:1 --cpus-per-task=4 --time="${S4_TIME:-100:00:00}" --mem="${S4_MEM:-64G}" --job-name="F2-${DATASET}-S4-${family}" --output="$OUT/s4_${family}_%j.out" --error="$OUT/s4_${family}_%j.err" --wrap="cd '$HERE' && python -u s4.py --family $family")
  j5=$(sbatch --parsable --account="${ACCOUNT:-PDE0114}" --gres=gpu:1 --cpus-per-task=4 --time="${S5_TIME:-48:00:00}" --mem="${S5_MEM:-32G}" --dependency="afterok:$j4" --job-name="F2-${DATASET}-S5-${family}" --output="$OUT/s5_${family}_%j.out" --error="$OUT/s5_${family}_%j.err" --wrap="cd '$HERE' && python -u s5.py --family $family")
  j6=$(sbatch --parsable --account="${ACCOUNT:-PDE0114}" --gres=gpu:1 --cpus-per-task=4 --time="${S6_TIME:-08:00:00}" --mem="${S6_MEM:-32G}" --dependency="afterok:$j5" --job-name="F2-${DATASET}-S6-${family}" --output="$OUT/s6_${family}_%j.out" --error="$OUT/s6_${family}_%j.err" --wrap="cd '$HERE' && python -u s6.py --family $family")
  echo "[SUBMITTED] $family: S4=$j4 -> S5=$j5 -> S6=$j6"
done
