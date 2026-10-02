#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fit_att_mul_add.data import load_current_data, subject_frame, subject_ids
from fit_att_mul_add.model_info import MODEL_INFOS, MODEL_ORDER
from fit_att_mul_add.simulation import simulate_subject

p = argparse.ArgumentParser(description="Posterior-conditioned resimulation for model-comparison classifier")
p.add_argument("--data", default="data/trial_eye.csv")
p.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
p.add_argument("--gen-start", type=int, default=1)
p.add_argument("--gen-end", type=int, default=20, help="inclusive")
p.add_argument("--subjects-per-file", type=int, default=500)
p.add_argument("--seed", type=int, default=9876)
p.add_argument("--max-rt", type=float, default=14.5)
p.add_argument("--output-root", default="outputs/resim_raw")
a = p.parse_args()

models = MODEL_ORDER if a.model == "all" else [a.model]
df = load_current_data(a.data)
ids = subject_ids(df)

for m_i, model in enumerate(models):
    npz_path = Path("outputs/posterior") / model / "posterior_draws.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Run 05_infer_parameters.py first: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    draws = z["draws"]
    pnames = list(z["param_names"])
    if draws.shape[0] != len(ids):
        raise ValueError(f"Posterior subject count {draws.shape[0]} != real subject count {len(ids)}")

    for gen in range(a.gen_start, a.gen_end + 1):
        rng = np.random.default_rng(a.seed + 100000 * m_i + gen)
        generated = []
        for _ in tqdm(range(a.subjects_per_file), desc=f"resim {model} gen{gen}"):
            i_subj = int(rng.integers(0, len(ids)))
            j_draw = int(rng.integers(0, draws.shape[1]))
            free = {str(name): float(draws[i_subj, j_draw, k]) for k, name in enumerate(pnames)}
            d = subject_frame(df, ids[i_subj])
            subj = simulate_subject(model, d, free, max_rt=a.max_rt)
            if len(subj["choice"]) <= 5:
                continue
            subj["posterior_subject_index"] = i_subj
            subj["posterior_draw_index"] = j_draw
            generated.append(subj)

        out = Path(a.output_root) / model / f"gen{gen}.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(generated, f, protocol=4)
        print(f"saved {out} ({len(generated)} subjects)")
