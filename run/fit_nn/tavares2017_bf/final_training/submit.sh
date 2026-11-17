#!/usr/bin/env bash
# One final-data S4 per classifier + dependent S5 recovery + S6 real inference.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BF_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$BF_DIR/../../.." && pwd)"
cd "$BF_DIR"

FAMILY="${1:-three}"
case "$FAMILY" in
  three) FAMILIES=(ddm race all) ;;
  ddm|race|all) FAMILIES=("$FAMILY") ;;
  *) echo "Usage: bash final_training/submit.sh [three|ddm|race|all]"; exit 2 ;;
esac

ACCOUNT="${ACCOUNT:-PDE0114}"
CPUS="${CPUS:-9}"
SIM_WORKERS="${SIM_WORKERS:-8}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
BATCHES="${BATCHES:-128}"
VALIDATION="${VALIDATION:-512}"
EPOCHS="${EPOCHS:-5000}"
EARLY_PATIENCE="${EARLY_PATIENCE:-20}"
LR_PATIENCE="${LR_PATIENCE:-6}"
RECOVERY_N="${RECOVERY_N:-500}"
INFERENCE_BATCH="${INFERENCE_BATCH:-4}"
TRAIN_MEM="${TRAIN_MEM:-64G}"
EVAL_MEM="${EVAL_MEM:-32G}"
TRAIN_TIME="${TRAIN_TIME:-100:00:00}"
EVAL_TIME="${EVAL_TIME:-100:00:00}"
FOLLOWUPS="${FOLLOWUPS:-1}"
SEED="${SEED:-20261030}"

if (( CPUS <= SIM_WORKERS )); then
  echo "ERROR: --cpus-per-task must be >= SIM_WORKERS + 1"
  exit 2
fi
if [[ "$FOLLOWUPS" != "1" && "$FOLLOWUPS" != "0" ]]; then
  echo "ERROR: FOLLOWUPS must be 0 or 1"; exit 2
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run this submitter on OSC login node, not inside sbatch"; exit 2
fi

# Check final CSV hash/QC before any job submission.
for fam in "${FAMILIES[@]}"; do
  python -u final_training/run.py --stage s4 --family "$fam" --dry-run \
    --epochs "$EPOCHS" --batches "$BATCHES" --batch-size "$TRAIN_BATCH" \
    --validation "$VALIDATION" --sim-workers "$SIM_WORKERS" \
    --early-patience "$EARLY_PATIENCE" --lr-patience "$LR_PATIENCE" \
    --seed "$SEED"
done

OUT="$ROOT/outputs/tavares2017_bf_final/g2_train_v1"
LOG="$OUT/slurm"
mkdir -p "$LOG"
EVAL_IDS=()

for fam in "${FAMILIES[@]}"; do
  base=(--parsable --account="$ACCOUNT" --gres=gpu:1)
  job="$(sbatch "${base[@]}" \
    --cpus-per-task="$CPUS" --time="$TRAIN_TIME" --mem="$TRAIN_MEM" \
    --job-name="TavFinal-S4-$fam" \
    --output="$LOG/S4_${fam}_%j.out" --error="$LOG/S4_${fam}_%j.err" \
    --wrap="cd \"$BF_DIR\" && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && python -u final_training/run.py --stage s4 --family $fam --mode formal --seed $SEED --epochs $EPOCHS --batches $BATCHES --batch-size $TRAIN_BATCH --validation $VALIDATION --sim-workers $SIM_WORKERS --early-patience $EARLY_PATIENCE --lr-patience $LR_PATIENCE")"
  job="${job%%;*}"
  echo "S4 $fam: job $job"
  if [[ "$FOLLOWUPS" == "1" ]]; then
    recover="$(sbatch "${base[@]}" \
      --cpus-per-task=2 --dependency="afterok:$job" --time="$EVAL_TIME" \
      --mem="$EVAL_MEM" --job-name="TavFinal-S5-$fam" \
      --output="$LOG/S5_${fam}_%j.out" --error="$LOG/S5_${fam}_%j.err" \
      --wrap="cd \"$BF_DIR\" && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 && python -u final_training/run.py --stage s5 --family $fam --mode formal --seed $SEED --n-per-model $RECOVERY_N --inference-batch $INFERENCE_BATCH")"
    recover="${recover%%;*}"
    real="$(sbatch "${base[@]}" \
      --cpus-per-task=2 --dependency="afterok:$job" --time="$EVAL_TIME" \
      --mem="$EVAL_MEM" --job-name="TavFinal-S6-$fam" \
      --output="$LOG/S6_${fam}_%j.out" --error="$LOG/S6_${fam}_%j.err" \
      --wrap="cd \"$BF_DIR\" && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 && python -u final_training/run.py --stage s6 --family $fam --mode formal --seed $SEED --inference-batch $INFERENCE_BATCH")"
    real="${real%%;*}"
    EVAL_IDS+=("$recover" "$real")
    echo "S5 $fam: job $recover (after S4 $job)"
    echo "S6 $fam: job $real (after S4 $job)"
  fi
done

if [[ "$FOLLOWUPS" == "1" ]]; then
  deps="$(IFS=:; echo "${EVAL_IDS[*]}")"
  report="$(sbatch --parsable --account="$ACCOUNT" \
    --dependency="afterok:$deps" --time=01:00:00 --mem=2G \
    --cpus-per-task=1 --job-name=TavFinal-Report \
    --output="$LOG/report_%j.out" --error="$LOG/report_%j.err" \
    --wrap="cd \"$BF_DIR\" && python -u final_training/report.py")"
  echo "SUMMARY: job $report (after all submitted S5/S6 jobs)"
fi
echo "[RESULT ROOT] $OUT"
echo "[LOG ROOT] $LOG"
