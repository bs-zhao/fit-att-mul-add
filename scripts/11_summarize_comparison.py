#!/usr/bin/env python
from pathlib import Path
import pandas as pd

from fit_att_mul_add.model_info import MODEL_LABELS, MODEL_ORDER

path = Path("outputs/model_comparison/real_model_probabilities.csv")
df = pd.read_csv(path)
summary = []
for m in MODEL_ORDER:
    summary.append({
        "model": m,
        "label": MODEL_LABELS[m],
        "mean_probability": df[m].mean(),
        "median_probability": df[m].median(),
        "n_winner": int((df["winner"] == m).sum()),
    })
out = pd.DataFrame(summary).sort_values("mean_probability", ascending=False)
out.to_csv("outputs/model_comparison/summary.csv", index=False)
print(out.to_string(index=False))
