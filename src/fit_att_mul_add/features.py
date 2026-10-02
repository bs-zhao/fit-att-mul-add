"""Feature construction shared by SBI regression and model classification."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .data import DT

EYE2 = np.eye(2, dtype=np.float32)


def fixation_summary(arr_pos, arr_du, dt: float = DT) -> np.ndarray:
    """Two-option analogue of the `fsmr=get_summary(...)` feature in adm-sbi."""
    pos = np.asarray(arr_pos, dtype=np.int64)
    du = np.asarray(arr_du, dtype=np.float64)
    if len(pos) == 0:
        raise ValueError("empty fixation sequence")
    total_steps = float(np.sum(du))
    total_steps = max(total_steps, 1.0)
    tt0 = float(np.sum(du[pos == 0]) * dt)
    tt1 = float(np.sum(du[pos == 1]) * dt)
    rt = total_steps * dt
    first = EYE2[int(pos[0])]
    last = EYE2[int(pos[-1])]
    return np.asarray([
        tt0, tt1, tt0 / rt, tt1 / rt,
        float(len(pos)), float(np.sum(pos == 0)), float(np.sum(pos == 1)),
        float(first[0]), float(first[1]), float(last[0]), float(last[1]),
    ], dtype=np.float32)


def raw_subject_to_feature(raw_subj: dict) -> dict:
    trialinfo, choice, rt, fsmr = [], [], [], []
    n_trials = len(raw_subj["choice"])
    for j in range(n_trials):
        pos = np.asarray(raw_subj["arr_pos"][j], dtype=np.int64)
        du = np.asarray(raw_subj["arr_du"][j], dtype=np.int64)
        if len(pos) == 0 or len(du) == 0:
            continue
        trialinfo.append(np.asarray(raw_subj["vs"][j], dtype=np.float32))
        choice.append(EYE2[int(raw_subj["choice"][j])])
        rt.append(float(raw_subj["rt"][j]))
        fsmr.append(fixation_summary(pos, du))

    return {
        "trialinfo": np.asarray(trialinfo, dtype=np.float32),
        "choice": np.asarray(choice, dtype=np.float32),
        "rt": np.asarray(rt, dtype=np.float32),
        "fsmr": np.asarray(fsmr, dtype=np.float32),
        "params": raw_subj.get("params"),
        "subj": raw_subj.get("subj"),
        "template_subj": raw_subj.get("template_subj", raw_subj.get("subj")),
    }


def save_feature_batches(raw_file: str | Path, out_dir: str | Path, batch_size: int = 500) -> list[Path]:
    raw_file = Path(raw_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with raw_file.open("rb") as f:
        raw = pickle.load(f)
    outputs = []
    for i in range(0, len(raw), batch_size):
        feats = []
        for subj in raw[i:i + batch_size]:
            fe = raw_subject_to_feature(subj)
            if len(fe["trialinfo"]) > 5:
                feats.append(fe)
        out = out_dir / f"{raw_file.stem}_batch{i}.pkl"
        with out.open("wb") as f:
            pickle.dump(feats, f, protocol=4)
        outputs.append(out)
    return outputs


def save_subject_feature(feature_subj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump([feature_subj], f, protocol=4)
    return path
