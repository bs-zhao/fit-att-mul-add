"""Simulation utilities using the original uploaded c_bifood model implementation."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import DT, subject_frame, subject_ids
from .model_info import complete_params, sample_free_params


def _simulator(model_name: str):
    try:
        from .simulators.c_bifood import sim_trial_aDDM, sim_trial_aRACE
    except ImportError as e:
        raise ImportError(
            "c_bifood is not compiled. From repository root run: "
            "python setup_simulator.py build_ext --inplace"
        ) from e
    return sim_trial_aDDM if model_name.startswith("aDDM") else sim_trial_aRACE


def truncate_fixations(arr_pos, arr_du, total_steps: int):
    pos = np.asarray(arr_pos, dtype=np.int64)
    du = np.asarray(arr_du, dtype=np.int64)
    if total_steps <= 0:
        total_steps = 1
    out_pos, out_du = [], []
    remaining = int(total_steps)
    for p, d in zip(pos, du):
        if remaining <= 0:
            break
        use = min(int(d), remaining)
        if use > 0:
            out_pos.append(int(p))
            out_du.append(use)
            remaining -= use
    if remaining > 0:
        out_pos.append(int(pos[-1]))
        out_du.append(remaining)
    return np.asarray(out_pos, dtype=np.int64), np.asarray(out_du, dtype=np.int64)


def simulate_trial(model_name: str, params: dict, values, arr_pos, arr_du, max_rt: float = 14.5):
    sim = _simulator(model_name)
    pos = np.asarray(arr_pos, dtype=np.int64).copy()
    du = np.asarray(arr_du, dtype=np.int64).copy()
    kwargs = dict(extend_last=500000, repeat=0, dt=DT, rcd=0, max_rt=max_rt)
    if model_name.startswith("aDDM"):
        (choice, rt, hit), _ = sim(params, np.asarray(values, dtype=np.float64), pos, du, **kwargs)
    else:
        (choice, rt, hit), _ = sim(params, np.asarray(values, dtype=np.float64), pos, du, rb=1, abslope=0, **kwargs)
    sim_pos, sim_du = truncate_fixations(arr_pos, arr_du, max(1, int(round(rt / DT))))
    return int(choice), float(rt), int(hit), sim_pos, sim_du


def simulate_subject(model_name: str, df_template, free_params: dict,
                     max_rt: float = 14.5, max_trial_retries: int = 10) -> dict:
    params = complete_params(model_name, free_params)
    arr_pos_out, arr_du_out, choice_out, rt_out, values_out = [], [], [], [], []
    for row in df_template.itertuples(index=False):
        values = np.asarray([row.v0, row.v1], dtype=np.float64)
        got_hit = False
        for _ in range(max_trial_retries):
            choice, rt, hit, sim_pos, sim_du = simulate_trial(
                model_name, params, values, row.arr_pos, row.arr_du, max_rt=max_rt
            )
            if hit:
                got_hit = True
                break
        if not got_hit:
            continue
        arr_pos_out.append(sim_pos)
        arr_du_out.append(sim_du)
        choice_out.append(choice)
        rt_out.append(rt)
        values_out.append(values)

    return {
        "subj": df_template.iloc[0]["subj"],
        "template_subj": df_template.iloc[0]["subj"],
        "arr_pos": np.asarray(arr_pos_out, dtype=object),
        "arr_du": np.asarray(arr_du_out, dtype=object),
        "choice": np.asarray(choice_out, dtype=np.int64),
        "rt": np.asarray(rt_out, dtype=np.float64),
        "vs": np.asarray(values_out, dtype=np.float64),
        "params": params,
        "free_params": dict(free_params),
    }


def simulate_training_file(model_name: str, df_real, out_path: str | Path,
                           n_subjects: int = 500, seed: int = 1234,
                           max_rt: float = 14.5) -> Path:
    rng = np.random.default_rng(seed)
    ids = subject_ids(df_real)
    generated = []
    for _ in tqdm(range(n_subjects), desc=f"simulate {model_name}"):
        template = ids[int(rng.integers(0, len(ids)))]
        df_template = subject_frame(df_real, template)
        for _attempt in range(50):
            free = sample_free_params(model_name, rng)
            subj = simulate_subject(model_name, df_template, free, max_rt=max_rt)
            if len(subj["choice"]) >= max(6, int(0.8 * len(df_template))):
                generated.append(subj)
                break
        else:
            raise RuntimeError(f"Could not generate a valid synthetic subject for {model_name}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(generated, f, protocol=4)
    return out_path
