#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data2param_flow import load_instance

from fit_att_mul_add.model_info import MODEL_INFOS, MODEL_ORDER, rescale_from_unit

p = argparse.ArgumentParser(description="Parameter-recovery diagnostic on simulated feature files")
p.add_argument("--model", choices=MODEL_ORDER, required=True)
p.add_argument("--n-files", type=int, default=3)
p.add_argument("--batch-size", type=int, default=64)
a = p.parse_args()

files = sorted((Path("outputs/training_features") / a.model).glob("gen*.pkl"))[-a.n_files:]
if not files:
    raise FileNotFoundError("No training feature files")
dec = load_instance(f"outputs/param_decoder/{a.model}/parameter_decoder.pkl")
dec.device = "cuda" if torch.cuda.is_available() else "cpu"
dec.batch_size = a.batch_size
pred_u, true_u = dec.predict([str(x) for x in files], key_param="params", post_dropout=True)
pred_u = pred_u.cpu().numpy()
true_u = true_u.cpu().numpy()
pred = np.asarray([rescale_from_unit(a.model, x) for x in pred_u])
true = np.asarray([rescale_from_unit(a.model, x) for x in true_u])

info = MODEL_INFOS[a.model]
rows = []
for k, name in enumerate(info["free_pnames"]):
    r = np.corrcoef(true[:, k], pred[:, k])[0, 1]
    mae = np.mean(np.abs(true[:, k] - pred[:, k]))
    rows.append({"parameter": name, "r": r, "mae": mae})
    print(f"{name}: r={r:.3f}, MAE={mae:.4f}")

out_dir = Path("outputs/diagnostics") / a.model
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_dir / "parameter_recovery_summary.csv", index=False)
wide = {}
for k, name in enumerate(info["free_pnames"]):
    wide[f"true_{name}"] = true[:, k]
    wide[f"pred_{name}"] = pred[:, k]
pd.DataFrame(wide).to_csv(out_dir / "parameter_recovery_rows.csv", index=False)
