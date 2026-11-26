#!/usr/bin/env bash
# Explicit stages only (does not prequeue S2/S3/S5/S6).
# Usage:
#   bash submit.sh s1 two
#   bash submit.sh s4 three
#   bash submit.sh s2 two
#   bash submit.sh s3 two
#   bash submit.sh s5 three
#   bash submit.sh s6 three
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DATASET="krajbich2010"
STAGE="${1:-}"
GROUP="${2:-}"
if [[ ! "$STAGE" =~ ^s[1-6]$ ]]; then
 echo "Usage: bash submit.sh <s1|s2|s3|s4|s5|s6> [two|all|three|ddm|race|combined|aDDM_t|aRACE_g]"; exit 2
fi
case "$STAGE" in
 s[123])
  case "$GROUP" in
   two) ITEMS=(aRACE_g aDDM_t);;
   all) ITEMS=(aDDM_1 aDDM_2 aDDM_t aDDM_g aRACE_1 aRACE_2 aRACE_t aRACE_g);;
   aDDM_1|aDDM_2|aDDM_t|aDDM_g|aRACE_1|aRACE_2|aRACE_t|aRACE_g) ITEMS=("$GROUP");;
   *) echo "For S1-S3 use two|all|<model>"; exit 2;;
  esac;;
 *)
  case "$GROUP" in
   three) ITEMS=(ddm race all);;
   ddm|race|all) ITEMS=("$GROUP");;
   *) echo "For S4-S6 use three|ddm|race|all"; exit 2;;
  esac;;
esac
ACCOUNT="${ACCOUNT:-PDE0114}"
TIME="${TIME:-100:00:00}"
MEM="${MEM:-64G}"
LOG="$ROOT/outputs/${DATASET}_final2/slurm"
mkdir -p "$LOG"
cd "$HERE"
for item in "${ITEMS[@]}"; do
 if [[ "$STAGE" =~ ^s[123]$ ]]; then
   ARGS=(--model "$item")
 else
   ARGS=(--family "$item")
 fi
 # Validate the SAME normalized model and data before submitting each job.
 python -u "${STAGE}.py" "${ARGS[@]}" --dry-run
 # S1/S4 need GPUs for training; recovery and inference also use GPUs.
 id="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 \
    --cpus-per-task=4 --mem="$MEM" --time="$TIME" \
    --job-name="F2-${DATASET}-${STAGE}-${item}" \
    --output="$LOG/${STAGE}_${item}_%j.out" \
    --error="$LOG/${STAGE}_${item}_%j.err" \
    --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u ${STAGE}.py ${ARGS[*]}")"
 echo "[SUBMITTED] $DATASET $STAGE $item job=$id"
done
echo "[OUTPUT] $ROOT/outputs/${DATASET}_final2"
echo "[LOGS] $LOG"
