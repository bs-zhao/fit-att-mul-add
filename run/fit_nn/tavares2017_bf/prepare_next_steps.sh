#!/usr/bin/env bash
# CPU-only preparation/audit; displays command for GPU baseline consistency test.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
python -u prepare_data.py
python -u next_steps/audit_original.py
python -m unittest discover -s next_steps/tests -v
echo "GPU-only baseline paired consistency test:"
echo "sbatch --account=PDE0114 --gres=gpu:1 --time=24:00:00 --mem=32G --wrap='cd $HERE && export KERAS_BACKEND=torch && python -u next_steps/consistency.py --n-per-model 32 --batch-size 4 --data both'"
