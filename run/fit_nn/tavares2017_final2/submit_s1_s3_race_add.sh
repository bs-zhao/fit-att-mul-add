#!/usr/bin/env bash
# Final2 aRACE_g only: S1 -> S2 -> S3 with Slurm afterok dependencies.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="$ROOT/outputs/tavares2017_final2"
LOG="$OUT/slurm"
mkdir -p "$LOG"
cd "$HERE"
if [[ -e "$OUT/aRACE_g/formal/best.keras" || -e "$OUT/aRACE_g/formal/metadata.json" ]]; then
  echo "Final2 aRACE_g S1 output exists; refusing duplicate training." >&2
  echo "If S1 is complete, run S2 and S3 separately after verifying prior output." >&2
  exit 2
fi
python -u s1.py --model aRACE_g --dry-run >/dev/null
job1=$(sbatch --parsable --account="${ACCOUNT:-PDE0114}" --gres=gpu:1 --cpus-per-task=4 --time="${S1_TIME:-100:00:00}" --mem=64G --job-name="F2-tavares2017-S1-RACEAdd" --output="$LOG/s1_aRACE_g_%j.out" --error="$LOG/s1_aRACE_g_%j.err" --wrap="cd '$HERE' && python -u s1.py --model aRACE_g")
job2=$(sbatch --parsable --account="${ACCOUNT:-PDE0114}" --gres=gpu:1 --cpus-per-task=4 --time="${S2_TIME:-08:00:00}" --mem=32G --dependency="afterok:$job1" --job-name="F2-tavares2017-S2-RACEAdd" --output="$LOG/s2_aRACE_g_%j.out" --error="$LOG/s2_aRACE_g_%j.err" --wrap="cd '$HERE' && python -u s2.py --model aRACE_g")
job3=$(sbatch --parsable --account="${ACCOUNT:-PDE0114}" --gres=gpu:1 --cpus-per-task=4 --time="${S3_TIME:-08:00:00}" --mem=32G --dependency="afterok:$job2" --job-name="F2-tavares2017-S3-RACEAdd" --output="$LOG/s3_aRACE_g_%j.out" --error="$LOG/s3_aRACE_g_%j.err" --wrap="cd '$HERE' && python -u s3.py --model aRACE_g")
printf 'Final2 %s aRACE_g: S1=%s -> S2=%s -> S3=%s\n' 'tavares2017' "$job1" "$job2" "$job3"
