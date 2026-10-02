"""Loading and standardizing the current food-choice dataset."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.001

_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_numeric_array(value, integer: bool) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        text = str(value)
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, (list, tuple, np.ndarray)):
                arr = np.asarray(obj)
            else:
                raise ValueError
        except Exception:
            pattern = _INT_RE if integer else _FLOAT_RE
            vals = pattern.findall(text)
            arr = np.asarray(vals)
    return arr.astype(np.int64 if integer else np.float64)


def load_current_data(path: str | Path) -> pd.DataFrame:
    """Load the exact columns used by the submitted behavioral fitting code.

    Required columns:
      subj, v0, v1, response, arr_ml3_left, arr_ml3_time

    `c_bifood` interprets vs as [right, left], therefore the historical code's
    `v0` is kept as the right value and `v1` as the left value.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Put trial_eye.csv under data/ (ignored by git)."
        )
    df = pd.read_csv(path)
    required = {"subj", "v0", "v1", "response", "arr_ml3_left", "arr_ml3_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["arr_pos"] = df["arr_ml3_left"].map(lambda x: _parse_numeric_array(x, True))
    df["arr_du"] = df["arr_ml3_time"].map(lambda x: _parse_numeric_array(x, False))

    cleaned_pos, cleaned_du = [], []
    for pos, du in zip(df["arr_pos"], df["arr_du"]):
        n = min(len(pos), len(du))
        pos = np.asarray(pos[:n], dtype=np.int64)
        du = np.asarray(np.rint(du[:n]), dtype=np.int64)
        keep = (pos >= 0) & (du > 0)
        pos, du = pos[keep], du[keep]
        if len(pos) == 0:
            cleaned_pos.append(np.asarray([], dtype=np.int64))
            cleaned_du.append(np.asarray([], dtype=np.int64))
            continue
        if not np.isin(pos, [0, 1]).all():
            bad = np.unique(pos[~np.isin(pos, [0, 1])])
            raise ValueError(f"Unexpected fixation codes {bad}; expected only 0/1 after removing -1.")
        cleaned_pos.append(pos)
        cleaned_du.append(du)

    df["arr_pos"] = cleaned_pos
    df["arr_du"] = cleaned_du
    df = df[df["arr_pos"].map(len) > 0].copy()
    df["v0"] = pd.to_numeric(df["v0"], errors="raise").astype(float)
    df["v1"] = pd.to_numeric(df["v1"], errors="raise").astype(float)
    df["response"] = pd.to_numeric(df["response"], errors="raise").astype(int)
    if not df["response"].isin([0, 1]).all():
        raise ValueError("response must be binary 0/1")
    df["rt_steps"] = df["arr_du"].map(lambda x: int(np.sum(x)))
    df["rt"] = df["rt_steps"] * DT
    return df.reset_index(drop=True)


def subject_ids(df: pd.DataFrame) -> list:
    return list(pd.unique(df["subj"]))


def subject_frame(df: pd.DataFrame, subj) -> pd.DataFrame:
    return df[df["subj"] == subj].reset_index(drop=True)
