#!/usr/bin/env bash
# Three independent S4s, followed by S5 diagnostics and S6 real subjects.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
INPUT="$HERE/../../../data/tavares2017/trial_eye.csv"
if [[ ! -s "$INPUT" ]]; then
  echo "Missing prepared data: $INPUT"
  echo "First run: python -u prepare_data.py"
  exit 2
fi
ACCOUNT="${ACCOUNT:-PDE0114}"
SIM_WORKERS="${SIM_WORKERS:-8}"
CPUS="${CPUS:-9}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
VALIDATION="${VALIDATION:-64}"
INFER_BATCH="${INFER_BATCH:-4}"
N_RECOVERY="${N_RECOVERY:-1000}"
TIME="${TIME:-100:00:00}"
MEM="${MEM:-32G}"
if (( CPUS <= SIM_WORKERS )); then
  echo "CPUS must be at least SIM_WORKERS+1"
  exit 2
fi

for family in ddm race all; do
  common=(--account="$ACCOUNT" --gres=gpu:1 --time="$TIME" --mem="$MEM")
  train="$(sbatch --parsable "${common[@]}" --cpus-per-task="$CPUS" \
      --job-name="Tav-G2-S4-$family" \
      --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && python -u experiments/parallel_normalized_value_prior_sweep.py --sim-workers $SIM_WORKERS --stage s4 --family $family --mode formal --batch-size $TRAIN_BATCH --validation $VALIDATION")"
  recover="$(sbatch --parsable "${common[@]}" --cpus-per-task=2 \
      --dependency="afterok:$train" --job-name="Tav-G2-S5-$family" \
      --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch && python -u experiments/normalized_value_prior_sweep.py --stage s5 --family $family --mode formal --n-per-model $N_RECOVERY --batch-size $INFER_BATCH")"
  real="$(sbatch --parsable "${common[@]}" --cpus-per-task=2 \
      --dependency="afterok:$train" --job-name="Tav-G2-S6-$family" \
      --wrap="cd \"$HERE\" && export KERAS_BACKEND=torch && python -u experiments/normalized_value_prior_sweep.py --stage s6 --family $family --mode formal --batch-size $INFER_BATCH")"
  echo "$family: S4=$train S5=$recover S6=$real (S5/S6 after S4 succeeds)"
done
echo "Review S5 recovery diagnostics before interpreting S6."
