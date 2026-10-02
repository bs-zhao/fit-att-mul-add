#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data2param_flow import load_instance

from fit_att_mul_add.model_info import MODEL_LABELS, MODEL_ORDER

p = argparse.ArgumentParser(description="Apply the trained 8-class model classifier to real participants")
p.add_argument("--batch-size", type=int, default=64)
a = p.parse_args()

manifest = pd.read_csv("outputs/real_features/manifest.csv")
files = manifest["file"].tolist()
path = Path("outputs/model_classifier/all8/parameter_decoder.pkl")
dec = load_instance(str(path))
dec.device = "cuda" if torch.cuda.is_available() else "cpu"
dec.batch_size = a.batch_size
logits = dec.predict(files, key_param=None, post_dropout=True)[0]
probs = torch.softmax(logits, dim=1).cpu().numpy()

out = manifest[["index", "subj", "n_trials"]].copy()
for j, model in enumerate(MODEL_ORDER):
    out[model] = probs[:, j]
out["winner"] = [MODEL_ORDER[i] for i in np.argmax(probs, axis=1)]
out["winner_label"] = out["winner"].map(MODEL_LABELS)
out_dir = Path("outputs/model_comparison")
out_dir.mkdir(parents=True, exist_ok=True)
out.to_csv(out_dir / "real_model_probabilities.csv", index=False)

print("mean model probabilities")
for j, model in enumerate(MODEL_ORDER):
    print(f"{MODEL_LABELS[model]:12s} {probs[:, j].mean():.4f}")
print("\nparticipant winners")
print(out["winner_label"].value_counts())
