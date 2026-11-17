#!/usr/bin/env bash
# S4 independent classifier jobs with optional dependent S5/S6.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PROFILE="${1:-}"
FAMILY="${2:-three}"
case "$PROFILE" in
 active_thomas_all|active_thomas_qc|active_relative_qc|full_thomas_all|full_thomas_qc|full_relative_qc) ;;
 *) echo "Usage: bash submit_variant.sh <profile> [ddm|race|all|three]"; exit 2;;
esac
case "$FAMILY" in
 ddm|race|all) FAMILIES=("$FAMILY") ;;
 three) FAMILIES=(ddm race all) ;;
 *) echo "Bad family $FAMILY"; exit 2 ;;
esac
DATA="$HERE/../../../data/tavares2017_v2/$PROFILE/trial_eye.csv"
if [[ ! -s "$DATA" ]]; then
 echo "Missing $DATA; first run python prepare_variants.py"; exit 2
fi
if [[ "$PROFILE" == full_* ]] && ! python -c 'import c_tavares_fulltimeline' >/dev/null 2>&1; then
 echo "Build extension first: python setup_cython.py build_ext --inplace"; exit 2
fi
ACCOUNT="${ACCOUNT:-PDE0114}"
SIM_WORKERS="${SIM_WORKERS:-8}"
CPUS="${CPUS:-9}"
BATCH_SIZE="${TRAIN_BATCH:-8}"
BATCHES="${BATCHES:-128}"
VALIDATION="${VALIDATION:-512}"
INFER_BATCH="${INFER_BATCH:-4}"
RECOVERY_N="${RECOVERY_N:-512}"
WALLTIME="${WALLTIME:-100:00:00}"
MEM="${MEM:-64G}"
FOLLOWUPS="${FOLLOWUPS:-1}"
if (( CPUS <= SIM_WORKERS )); then echo "CPUS must exceed SIM_WORKERS"; exit 2; fi
for fam in "${FAMILIES[@]}"; do
 train=$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 \
  --cpus-per-task="$CPUS" --time="$WALLTIME" --mem="$MEM" \
  --job-name="Tav2-$PROFILE-$fam-S4" \
  --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u experiments/parallel_normalized_value_prior_sweep.py --sim-workers $SIM_WORKERS --stage s4 --family $fam --mode formal --input \"$DATA\" --batches $BATCHES --batch-size $BATCH_SIZE --validation $VALIDATION")
 train="${train%%;*}"
 echo "$PROFILE / $fam: S4 job $train"
 if [[ "$FOLLOWUPS" == "1" ]]; then
  for stage in s5 s6; do
   if [[ "$stage" == "s5" ]]; then
    extra="--n-per-model $RECOVERY_N --batch-size $INFER_BATCH"
   else
    extra="--batch-size $INFER_BATCH"
   fi
   after=$(sbatch --parsable --account="$ACCOUNT" --gres=gpu:1 \
    --cpus-per-task=2 --time="$WALLTIME" --mem="$MEM" \
    --dependency="afterok:$train" --job-name="Tav2-$PROFILE-$fam-$stage" \
    --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u experiments/normalized_value_prior_sweep.py --stage $stage --family $fam --mode formal --input \"$DATA\" $extra")
   echo "$PROFILE / $fam: $stage $after queued after S4 $train"
  done
 fi
done
