#!/usr/bin/env python
import argparse
from pathlib import Path

from fit_att_mul_add.features import save_feature_batches
from fit_att_mul_add.model_info import MODEL_ORDER

p = argparse.ArgumentParser()
p.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
p.add_argument("--batch-size", type=int, default=500)
a = p.parse_args()
models = MODEL_ORDER if a.model == "all" else [a.model]

for model in models:
    src = Path("outputs/resim_raw") / model
    dst = Path("outputs/classifier_features") / model
    files = sorted(src.glob("gen*.pkl"))
    if not files:
        raise FileNotFoundError(f"No posterior resimulation files in {src}")
    for f in files:
        outs = save_feature_batches(f, dst, a.batch_size)
        print(f"{model} {f.name}: wrote {len(outs)} feature batch(es)")
