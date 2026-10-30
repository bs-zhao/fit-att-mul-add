#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python -c 'import sys; assert (3,12) <= sys.version_info[:2] < (3,14), "Use Python 3.12 or 3.13"'
python -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())'
python -m pip install -r "$here/requirements.txt"
python -m pip install --no-build-isolation -e "$here/../../../packages/c_bifood"
KERAS_BACKEND=torch python -c 'import bayesflow, keras, c_bifood; print("BayesFlow", bayesflow.__version__, "backend", keras.backend.backend())'
