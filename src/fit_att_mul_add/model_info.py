"""The eight behavioral models used in the submitted manuscript.

The parameterization follows the uploaded code. In particular, the historical
multiplicative training range is theta in [0.1, 1.0]. The endpoint models
AttOnly (theta=0) and NoAtt (theta=1) are kept as separate candidate models.

All simulator defaults are filled explicitly so every model can be passed to
`c_bifood` without relying on stale model_info variants.
"""

from __future__ import annotations

from copy import deepcopy

DEFAULT_PARAMS = {
    "theta": 1.0,
    "gamma": 0.0,
    "d": 1.0,
    "a": 2.0,
    "lam": -1.0,
    "k": 1.0,
    "s": 1.0,
    "ndt": 0.0,
    "b": 0.0,
    "var_v": 0.0,
    "b0_cslope": 0.0,
    "d_cslope": 0.0,
    "std_cslope": 0.0,
    "ov_cslope": 0.0,
}

MODEL_INFOS = {
    "aDDM_1": {
        "architecture": "ddm",
        "bias_form": "noatt",
        "free_pnames": ["d", "a"],
        "free_pranges": [(0.1, 5.0), (1.0, 5.0)],
        "fixed": {"theta": 1.0, "gamma": 0.0},
    },
    "aDDM_2": {
        "architecture": "ddm",
        "bias_form": "attonly",
        "free_pnames": ["d", "a"],
        "free_pranges": [(0.1, 5.0), (1.0, 5.0)],
        "fixed": {"theta": 0.0, "gamma": 0.0},
    },
    "aDDM_t": {
        "architecture": "ddm",
        "bias_form": "mul",
        "free_pnames": ["d", "a", "theta"],
        "free_pranges": [(0.1, 5.0), (1.0, 5.0), (0.1, 1.0)],
        "fixed": {"gamma": 0.0},
    },
    "aDDM_g": {
        "architecture": "ddm",
        "bias_form": "add",
        "free_pnames": ["d", "a", "gamma"],
        "free_pranges": [(0.1, 5.0), (1.0, 5.0), (0.0, 1.0)],
        "fixed": {"theta": 1.0},
    },
    "aRACE_1": {
        "architecture": "race",
        "bias_form": "noatt",
        "free_pnames": ["d", "a"],
        "free_pranges": [(0.1, 8.0), (0.5, 5.0)],
        "fixed": {"theta": 1.0, "gamma": 0.0, "b": 0.0},
    },
    "aRACE_2": {
        "architecture": "race",
        "bias_form": "attonly",
        "free_pnames": ["d", "a"],
        "free_pranges": [(0.1, 8.0), (0.5, 5.0)],
        "fixed": {"theta": 0.0, "gamma": 0.0, "b": 0.0},
    },
    "aRACE_t": {
        "architecture": "race",
        "bias_form": "mul",
        "free_pnames": ["d", "a", "theta"],
        "free_pranges": [(0.1, 8.0), (0.5, 5.0), (0.1, 1.0)],
        "fixed": {"gamma": 0.0, "b": 0.0},
    },
    "aRACE_g": {
        "architecture": "race",
        "bias_form": "add",
        "free_pnames": ["d", "a", "gamma"],
        "free_pranges": [(0.1, 8.0), (0.5, 5.0), (0.0, 1.0)],
        "fixed": {"theta": 1.0, "b": 0.0},
    },
}

MODEL_ORDER = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]

MODEL_LABELS = {
    "aDDM_1": "DDM-NoAtt",
    "aDDM_2": "DDM-AttOnly",
    "aDDM_t": "DDM-Mul",
    "aDDM_g": "DDM-Add",
    "aRACE_1": "ACC-NoAtt",
    "aRACE_2": "ACC-AttOnly",
    "aRACE_t": "ACC-Mul",
    "aRACE_g": "ACC-Add",
}


def get_model_info(model_name: str) -> dict:
    if model_name not in MODEL_INFOS:
        raise KeyError(f"Unknown model {model_name!r}. Choices: {MODEL_ORDER}")
    return deepcopy(MODEL_INFOS[model_name])


def complete_params(model_name: str, free_params: dict[str, float]) -> dict[str, float]:
    info = get_model_info(model_name)
    out = dict(DEFAULT_PARAMS)
    out.update(info["fixed"])
    out.update({k: float(v) for k, v in free_params.items()})
    return out


def sample_free_params(model_name: str, rng) -> dict[str, float]:
    info = get_model_info(model_name)
    return {
        name: float(rng.uniform(lo, hi))
        for name, (lo, hi) in zip(info["free_pnames"], info["free_pranges"])
    }


def rescale_from_unit(model_name: str, x):
    """Map data2param-flow normalized [0,1] output to native parameter ranges."""
    info = get_model_info(model_name)
    return [
        float(v) * (hi - lo) + lo
        for v, (lo, hi) in zip(x, info["free_pranges"])
    ]
