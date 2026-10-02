#!/usr/bin/env python
import argparse
from pathlib import Path

from fit_att_mul_add.features import save_feature_batches
from fit_att_mul_add.model_info import MODEL_ORDER

p = argparse.ArgumentParser()
p.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
p.add_argument("--batch-size", type=int, default=500)
p.add_argument("--input-root", default="outputs/resim_raw")
p.add_argument("--output-root", default="outputs/classifier_features")
a = p.parse_args()
models = MODEL_ORDER if a.model == "all" else [a.model]

for model in models:
    src = Path(a.input_root) / model
    dst = Path(a.output_root) / model
    files = sorted(src.glob("gen*.pkl"))
    if not files:
        raise FileNotFoundError(f"No posterior resimulation files in {src}")
    for f in files:
        outs = save_feature_batches(f, dst, a.batch_size)
        print(f"{model} {f.name}: wrote {len(outs)} feature batch(es)")
