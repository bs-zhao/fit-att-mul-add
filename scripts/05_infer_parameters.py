#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data2param_flow import load_instance

from fit_att_mul_add.model_info import MODEL_INFOS, MODEL_ORDER, rescale_from_unit

p = argparse.ArgumentParser(description="Infer participant-specific flow posteriors")
p.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
p.add_argument("--posterior-samples", type=int, default=500)
p.add_argument("--batch-size", type=int, default=64)
a = p.parse_args()

manifest = pd.read_csv("outputs/real_features/manifest.csv")
files = manifest["file"].tolist()
models = MODEL_ORDER if a.model == "all" else [a.model]
for model in models:
    path = Path("outputs/param_decoder") / model / "parameter_decoder.pkl"
    dec = load_instance(str(path))
    dec.device = "cuda" if torch.cuda.is_available() else "cpu"
    dec.batch_size = a.batch_size
    center_unit = dec.predict(files, key_param=None, post_dropout=True)[0].cpu().numpy()
    draws_unit = dec.sample_posterior(files, key_param=None, num_samples=a.posterior_samples).cpu().numpy()

    center = np.asarray([rescale_from_unit(model, row) for row in center_unit])
    draws = np.empty_like(draws_unit, dtype=float)
    for i in range(draws_unit.shape[0]):
        for j in range(draws_unit.shape[1]):
            draws[i, j] = rescale_from_unit(model, draws_unit[i, j])

    info = MODEL_INFOS[model]
    df = manifest[["index", "subj", "n_trials"]].copy()
    for k, name in enumerate(info["free_pnames"]):
        df[name] = center[:, k]
        df[f"{name}_sd"] = draws[:, :, k].std(axis=1)
    out_dir = Path("outputs/posterior") / model
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "posterior_center.csv", index=False)
    np.savez_compressed(
        out_dir / "posterior_draws.npz",
        draws=draws,
        center=center,
        subjects=manifest["subj"].astype(str).to_numpy(),
        param_names=np.asarray(info["free_pnames"], dtype=object),
    )
    print(f"saved posterior for {model}: {out_dir}")
