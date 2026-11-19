#!/usr/bin/env bash
# Submit ONLY 2x2 cross S6: old checkpoint x new data and vice versa.
# Old/old and new/new S6 cells are read from existing outputs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BF_DIR="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$BF_DIR/../../.." && pwd)"
cd "$BF_DIR"
CASE="${1:-both}"
case "$CASE" in
  both) FAMILIES=(ddm race) ;;
  three) FAMILIES=(ddm race all) ;;
  ddm|race|all) FAMILIES=("$CASE") ;;
  *) echo "Usage: bash final_training/submit_cross.sh [both|three|ddm|race|all]"; exit 2 ;;
esac
ACCOUNT="${ACCOUNT:-PDE0114}"
TIME="${TIME:-08:00:00}"
MEM="${MEM:-32G}"
BATCH="${BATCH:-4}"
CPUS="${CPUS:-2}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this submitter from an OSC login node, not within sbatch."; exit 2
fi
if (( BATCH < 1 || CPUS < 1 )); then
  echo "BATCH and CPUS must be positive integers."; exit 2
fi

echo "[CHECK] Metadata, original CSV hashes, G2 priors and output locations"
for family in "${FAMILIES[@]}"; do
  python -u final_training/cross_inference.py \
    --family "$family" --checkpoint old --data new --dry-run
  python -u final_training/cross_inference.py \
    --family "$family" --checkpoint new --data old --dry-run
done

OUT="$ROOT/outputs/tavares2017_bf_final/g2_train_v1/cross_inference"
LOG="$OUT/slurm"
mkdir -p "$LOG"
JOBS=()
for family in "${FAMILIES[@]}"; do
  for spec in old:new new:old; do
    checkpoint="${spec%%:*}"
    dataset="${spec##*:}"
    job="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 \
      --cpus-per-task="$CPUS" --time="$TIME" --mem="$MEM" \
      --job-name="TavCross-${family}-${checkpoint}${dataset}" \
      --output="$LOG/${family}_${checkpoint}ck_${dataset}data_%j.out" \
      --error="$LOG/${family}_${checkpoint}ck_${dataset}data_%j.err" \
      --wrap="cd \"$BF_DIR\" && export KERAS_BACKEND=torch MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u final_training/cross_inference.py --family $family --checkpoint $checkpoint --data $dataset --batch-size $BATCH")"
    job="${job%%;*}"
    JOBS+=("$job")
    echo "[GPU] $family $checkpoint checkpoint x $dataset data => $job"
  done
done
deps="$(IFS=:; echo "${JOBS[*]}")"
report="$(sbatch --parsable --account="$ACCOUNT" --time=00:30:00 --mem=2G \
  --cpus-per-task=1 --dependency="afterok:$deps" \
  --job-name=TavCross-Report \
  --output="$LOG/report_%j.out" --error="$LOG/report_%j.err" \
  --wrap="cd \"$BF_DIR\" && python -u final_training/cross_report.py --families ${FAMILIES[*]}")"
echo "[CPU SUMMARY] $report (after all cross S6 jobs succeed)"
echo "[OUTPUT] $OUT"
echo "[LOGS] $LOG"
