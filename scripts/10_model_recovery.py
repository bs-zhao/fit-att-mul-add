#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data2param_flow import load_instance
from sklearn.metrics import confusion_matrix

from fit_att_mul_add.model_info import MODEL_LABELS, MODEL_ORDER

p = argparse.ArgumentParser(description="Classifier model-recovery matrix on regenerated data")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--feature-root", default="outputs/recovery_features",
               help="Independent recovery-test features; must not overlap classifier training files")
p.add_argument("--max-files-per-model", type=int, default=-1,
               help="Optional cap; -1 uses all independent recovery files")
a = p.parse_args()

path = Path("outputs/model_classifier/all8/parameter_decoder.pkl")
dec = load_instance(str(path))
dec.device = "cuda" if torch.cuda.is_available() else "cpu"
dec.batch_size = a.batch_size

y_true, y_pred, rows = [], [], []
for i, model in enumerate(MODEL_ORDER):
    files = sorted((Path(a.feature_root) / model).glob("gen*.pkl"))
    if a.max_files_per_model > 0:
        files = files[-a.max_files_per_model:]
    if not files:
        raise FileNotFoundError(f"No independent recovery features for {model} in {a.feature_root}")
    logits = dec.predict([str(x) for x in files], key_param=None, post_dropout=True)[0]
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = np.argmax(probs, axis=1)
    y_true.extend([i] * len(pred))
    y_pred.extend(pred.tolist())
    for r in probs:
        rows.append({"true_model": model, **{m: r[j] for j, m in enumerate(MODEL_ORDER)}})

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(MODEL_ORDER)))
cm_prop = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
out_dir = Path("outputs/model_comparison")
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(cm, index=MODEL_ORDER, columns=MODEL_ORDER).to_csv(out_dir / "recovery_counts.csv")
pd.DataFrame(cm_prop, index=MODEL_ORDER, columns=MODEL_ORDER).to_csv(out_dir / "recovery_row_proportions.csv")
pd.DataFrame(rows).to_csv(out_dir / "recovery_probabilities.csv", index=False)

print("model recovery (row-normalized)")
print(pd.DataFrame(cm_prop, index=[MODEL_LABELS[m] for m in MODEL_ORDER], columns=[MODEL_LABELS[m] for m in MODEL_ORDER]).round(3))
print(f"overall accuracy={np.mean(np.asarray(y_true)==np.asarray(y_pred)):.4f}")
