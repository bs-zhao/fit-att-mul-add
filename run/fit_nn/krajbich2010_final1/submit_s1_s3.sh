#!/usr/bin/env bash
# Final1 eight independent parameter decoders: S1 -> S2 -> S3.
# Run once per dataset. All jobs use the same Final1 priors / value scaling.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DATASET="krajbich2010"
OUTPUT="$ROOT/outputs/${DATASET}_final1"
LOG="$OUTPUT/slurm"
ACCOUNT="${ACCOUNT:-PDE0114}"
TRAIN_TIME="${TRAIN_TIME:-100:00:00}"
RECOVERY_TIME="${RECOVERY_TIME:-08:00:00}"
DECODE_TIME="${DECODE_TIME:-04:00:00}"
MODELS=(aDDM_1 aDDM_2 aDDM_t aDDM_g aRACE_1 aRACE_2 aRACE_t aRACE_g)
command -v sbatch >/dev/null || { echo "Missing sbatch" >&2; exit 1; }
cd "$HERE"
# Preflight the full selection before scheduling ANY job.
for model in "${MODELS[@]}"; do
  if [[ -e "$OUTPUT/$model/formal/best.keras" || -e "$OUTPUT/$model/formal/metadata.json" ]]; then
    echo "[ERROR] Existing S1 checkpoint or metadata for $model; no jobs submitted." >&2
    echo "Use 'bash submit.sh s2 all', then 'bash submit.sh s3 all' after recovery." >&2
    exit 2
  fi
  python -u s1.py --model "$model" --mode formal --dry-run >/dev/null
done
mkdir -p "$LOG"

submit() {
  local stage="$1" model="$2" time="$3" mem="$4" predecessor="${5:-}"
  local dep=()
  if [[ -n "$predecessor" ]]; then dep=(--dependency="afterok:$predecessor"); fi
  sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 --cpus-per-task=4 \
    --time="$time" --mem="$mem" \
    --job-name="F1-krajbich2010-$stage-$model" \
    --output="$LOG/${stage}_${model}_%j.out" \
    --error="$LOG/${stage}_${model}_%j.err" \
    "${dep[@]}" \
    --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u ${stage}.py --model ${model} --mode formal"
}
for model in "${MODELS[@]}"; do
  s1="$(submit s1 "$model" "$TRAIN_TIME" 64G)"
  s2="$(submit s2 "$model" "$RECOVERY_TIME" 32G "$s1")"
  s3="$(submit s3 "$model" "$DECODE_TIME" 32G "$s2")"
  printf '[SUBMITTED] %-9s S1=%s -> S2=%s -> S3=%s\n' "$model" "$s1" "$s2" "$s3"
done
echo "[OUTPUT] $OUTPUT"
echo "[LOGS] $LOG"
echo "[NOTE] 24 scheduled jobs, only the 8 S1 tasks are initially eligible to run."
