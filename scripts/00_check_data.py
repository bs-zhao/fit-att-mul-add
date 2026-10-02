#!/usr/bin/env python
import argparse
from fit_att_mul_add.data import load_current_data, subject_ids

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/trial_eye.csv")
a = p.parse_args()

df = load_current_data(a.data)
print(f"rows={len(df)} subjects={len(subject_ids(df))}")
print(df[["subj", "v0", "v1", "response", "rt_steps", "rt"]].describe(include="all"))
print("trials per subject:")
print(df.groupby("subj").size().describe())
