#!/usr/bin/env python
import argparse
from pathlib import Path

from fit_att_mul_add.data import load_current_data
from fit_att_mul_add.model_info import MODEL_ORDER
from fit_att_mul_add.simulation import simulate_training_file

p = argparse.ArgumentParser(description="Prior-predictive simulation for parameter-decoder training")
p.add_argument("--data", default="data/trial_eye.csv")
p.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
p.add_argument("--gen-start", type=int, default=1)
p.add_argument("--gen-end", type=int, default=50, help="inclusive")
p.add_argument("--subjects-per-file", type=int, default=500)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--max-rt", type=float, default=14.5)
a = p.parse_args()

models = MODEL_ORDER if a.model == "all" else [a.model]
df = load_current_data(a.data)
for m_i, model in enumerate(models):
    for gen in range(a.gen_start, a.gen_end + 1):
        out = Path("outputs/sim_training") / model / f"gen{gen}.pkl"
        seed = a.seed + 100000 * m_i + gen
        print(f"[{model}] gen{gen} -> {out}")
        simulate_training_file(model, df, out, a.subjects_per_file, seed, a.max_rt)
