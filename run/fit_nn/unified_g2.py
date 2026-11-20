"""Shared model specification and value normalization for the three S1–S6 workflows."""

from copy import deepcopy
from pathlib import Path
from runpy import run_path

import numpy as np


PROFILE = "unified_g2_v1"
PRIORS = {
    "d": (1.0, 50.0),
    "a": (0.5, 5.0),
    "theta": (0.0, 1.0),
    "gamma": (0.0, 2.0),
}
BASE_MODELS = run_path(
    str(Path(__file__).resolve().parent / "krajbich2010/tools/model_info.py")
)["model_infos"]


def model_specifications():
    """Copy the shared eight models and replace their free parameter bounds."""
    models = deepcopy(BASE_MODELS)
    for info in models.values():
        info["free_pranges"] = [PRIORS[name] for name in info["free_pnames"]]
        info["free_pranges_ml"] = [PRIORS[name] for name in info["free_pnames_ml"]]
    return models


def normalize_values(frame):
    """Divide both value columns by the same dataset-wide maximum in place."""
    columns = ["v0", "v1"]
    values = frame[columns].to_numpy(dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Dataset must contain finite values for both options")
    maximum = float(values.max())
    if maximum <= 0:
        raise ValueError(f"Dataset maximum must be positive, got {maximum}")
    for column, scaled in zip(columns, (values / maximum).T):
        frame[column] = scaled
    return float(values.min()), maximum


def checkpoint_spec(experiment, names):
    """Describe the generating model and data scale needed to reuse a checkpoint."""
    return {
        "profile": PROFILE,
        "value_normalization": {
            "method": "dataset_max",
            "columns": ["v0", "v1"],
            "raw_min": experiment.value_min,
            "raw_max": experiment.value_max,
        },
        "free_priors": {
            name: {key: list(bounds) for key, bounds in zip(
                experiment.models[name]["free_pnames"],
                experiment.models[name]["free_pranges"],
            )}
            for name in names
        },
        "fixed_parameters": {
            name: dict(zip(
                experiment.models[name]["fixed_pnames"],
                experiment.models[name]["fixed_pvalues"],
            ))
            for name in names
        },
        "mul_model": {name: experiment.models[name]["mul_model"] for name in names},
    }


def validate_checkpoint(metadata, experiment, names, feature_names, folder):
    if metadata.get("prior_spec") != checkpoint_spec(experiment, names):
        raise ValueError(f"Checkpoint does not match {PROFILE}: {folder}")
    if (
        metadata.get("features") != feature_names
        or metadata.get("feature_scale") != experiment.scale.tolist()
        or metadata.get("max_trials") != experiment.max_trials
    ):
        raise ValueError(f"Checkpoint feature specification does not match: {folder}")
