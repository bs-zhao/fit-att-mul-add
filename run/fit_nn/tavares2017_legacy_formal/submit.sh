#!/usr/bin/env bash
# Formal original Tavares G2 experiment; three independent S4, S5, S6.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$HERE"
WHICH="${1:-three}"
case "$WHICH" in
 three) FAMILIES=(ddm race all) ;;
 ddm|race|all) FAMILIES=("$WHICH") ;;
 *) echo "Usage: bash submit.sh [three|ddm|race|all]"; exit 2 ;;
esac
ACCOUNT="${ACCOUNT:-PDE0114}"
CPUS="${CPUS:-9}"
SIM_WORKERS="${SIM_WORKERS:-8}"
MEM="${MEM:-64G}"
EVAL_MEM="${EVAL_MEM:-32G}"
TIME="${TIME:-100:00:00}"
EVAL_TIME="${EVAL_TIME:-48:00:00}"
BATCHES="${BATCHES:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VALIDATION="${VALIDATION:-512}"
EPOCHS="${EPOCHS:-5000}"
RECOVERY="${RECOVERY:-2000}"
SEED="${SEED:-20261030}"
if (( CPUS <= SIM_WORKERS )); then echo "CPUS must exceed SIM_WORKERS"; exit 2; fi

for fam in "${FAMILIES[@]}"; do
 python -u s4.py --family "$fam" --dry-run \
  --batches "$BATCHES" --batch-size "$BATCH_SIZE" --validation "$VALIDATION" \
  --epochs "$EPOCHS" --sim-workers "$SIM_WORKERS"
done

LOG="$ROOT/outputs/tavares2017_legacy_formal/v1/slurm"
mkdir -p "$LOG"
for fam in "${FAMILIES[@]}"; do
 s4="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 \
   --cpus-per-task="$CPUS" --time="$TIME" --mem="$MEM" \
   --job-name="TavOldFormalS4-$fam" \
   --output="$LOG/s4_${fam}_%j.out" --error="$LOG/s4_${fam}_%j.err" \
   --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u s4.py --family $fam --seed $SEED --epochs $EPOCHS --batches $BATCHES --batch-size $BATCH_SIZE --validation $VALIDATION --sim-workers $SIM_WORKERS")"
 s4="${s4%%;*}"
 s5="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 \
   --cpus-per-task=2 --time="$EVAL_TIME" --mem="$EVAL_MEM" \
   --dependency="afterok:$s4" --job-name="TavOldFormalS5-$fam" \
   --output="$LOG/s5_${fam}_%j.out" --error="$LOG/s5_${fam}_%j.err" \
   --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch && python -u s5.py --family $fam --seed $SEED --n-per-model $RECOVERY")"
 s5="${s5%%;*}"
 s6="$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 \
   --cpus-per-task=2 --time="$EVAL_TIME" --mem="$EVAL_MEM" \
   --dependency="afterok:$s4" --job-name="TavOldFormalS6-$fam" \
   --output="$LOG/s6_${fam}_%j.out" --error="$LOG/s6_${fam}_%j.err" \
   --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch && python -u s6.py --family $fam --seed $SEED")"
 s6="${s6%%;*}"
 echo "$fam: S4=$s4 S5=$s5 S6=$s6 (after S4)"
done
echo "[OUTPUT ROOT] $ROOT/outputs/tavares2017_legacy_formal/v1"
echo "[LOG ROOT] $LOG"
