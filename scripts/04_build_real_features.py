#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from fit_att_mul_add.data import load_current_data, subject_ids, subject_frame
from fit_att_mul_add.features import raw_subject_to_feature, save_subject_feature

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/trial_eye.csv")
a = p.parse_args()

df = load_current_data(a.data)
out_dir = Path("outputs/real_features/subjs")
out_dir.mkdir(parents=True, exist_ok=True)
manifest = []
for i, subj in enumerate(subject_ids(df)):
    d = subject_frame(df, subj)
    raw = {
        "subj": subj,
        "template_subj": subj,
        "arr_pos": np.asarray(d.arr_pos.tolist(), dtype=object),
        "arr_du": np.asarray(d.arr_du.tolist(), dtype=object),
        "choice": d.response.to_numpy(dtype=np.int64),
        "rt": d.rt.to_numpy(dtype=float),
        "vs": np.asarray([[r.v0, r.v1] for r in d.itertuples(index=False)], dtype=float),
        "params": None,
    }
    fe = raw_subject_to_feature(raw)
    path = out_dir / f"subj_{i:03d}.pkl"
    save_subject_feature(fe, path)
    manifest.append({"index": i, "subj": subj, "n_trials": len(fe["choice"]), "file": str(path)})

pd.DataFrame(manifest).to_csv("outputs/real_features/manifest.csv", index=False)
print(f"saved {len(manifest)} subject feature files")
