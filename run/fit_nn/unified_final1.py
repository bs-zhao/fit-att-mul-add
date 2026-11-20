"""Unified Final1 model/normalization contract for three eye-tracking datasets.

This module intentionally does NOT alter the legacy G2 module on disk.
Each Final1 entry point installs the new settings on the G2 module
object *in its own Python process* before any S1–S6 stage is imported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROFILE = "unified_final1_v1"
PRIORS = {
    "d": (0.1, 5.0),
    "a": (0.5, 10.0),
    "theta": (0.0, 1.0),
    "gamma": (0.0, 10.0),
}
DATASETS = ("food_equal", "krajbich2010", "tavares2017")
MODELS = (
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
)
FAMILIES = {
    "ddm": MODELS[:4],
    "race": MODELS[4:],
    "all": MODELS,
}
VALUE_METHOD = "dataset_max"

def configuration():
    return {
        "profile": PROFILE,
        "priors": {k: list(v) for k, v in PRIORS.items()},
        "models": list(MODELS),
        "value_normalization": {
            "method": VALUE_METHOD,
            "value_columns": ["v0", "v1"],
            "scope": "one global maximum for both columns, all subjects/trials, per dataset",
            "formula": "v_normalized = v_original / max(v0,v1)",
            "negative_values": "preserved (not min-max rescaled)",
        },
        "shared_fixed_parameters": {"ndt": 0, "s": 1, "lam": -1},
    }

def config_hash():
    payload=json.dumps(configuration(),sort_keys=True,separators=(",",":")).encode("utf8")
    return hashlib.sha256(payload).hexdigest()

def validate_models(models):
    if tuple(models) != MODELS:
        raise AssertionError(f"Model names/order mismatch: {list(models)}")
    canonical_fixed = {}
    for name in MODELS:
        spec=models[name]
        expected_names=("d","a", "theta" if name.endswith("_t") else "gamma") if name.endswith(("_t","_g")) else ("d","a")
        for pfield,bfield in (("free_pnames","free_pranges"),
                              ("free_pnames_ml","free_pranges_ml")):
            if tuple(spec[pfield]) != expected_names:
                raise AssertionError(f"{name}: free parameter set differs ({pfield})")
            bounds=dict(zip(spec[pfield],spec[bfield]))
            for param, prior in PRIORS.items():
                if param in bounds and tuple(map(float,bounds[param])) != prior:
                    raise AssertionError(f"{name}: {param} prior mismatch ({bfield})")
        fixed=dict(zip(spec["fixed_pnames"],spec["fixed_pvalues"]))
        for p,val in configuration()["shared_fixed_parameters"].items():
            if fixed.get(p)!=val:
                raise AssertionError(f"{name}: fixed {p} != {val}")
        if name.startswith("aRACE") and fixed.get("b")!=0:
            raise AssertionError(f"{name}: RACE b must be 0")
        if name.endswith("_1") and fixed.get("theta")!=1:
            raise AssertionError(f"{name}: NoAtt theta must be 1")
        if name.endswith("_2") and fixed.get("theta")!=0:
            raise AssertionError(f"{name}: AttOnly theta must be 0")
        if name.endswith("_g") and fixed.get("theta")!=1:
            raise AssertionError(f"{name}: Add theta must be 1")
        if not name.endswith("_g") and fixed.get("gamma")!=0:
            raise AssertionError(f"{name}: non-Add gamma must be 0")
        fixed_key=(name.replace("aDDM","",1).replace("aRACE","",1))
        if fixed_key in canonical_fixed and canonical_fixed[fixed_key] != {
            k:v for k,v in fixed.items() if k!="b"
        }:
            raise AssertionError(f"{name}: family fixed parameters differ")
        canonical_fixed[fixed_key]={k:v for k,v in fixed.items() if k!="b"}
    return True

def install(g2_module):
    """Patch only per-process imported shared module, then validate all eight."""
    g2_module.PROFILE=PROFILE
    g2_module.PRIORS=dict(PRIORS)
    models=g2_module.model_specifications()
    validate_models(models)
    return models

def audit_experiment(ex):
    if ex.value_normalization != VALUE_METHOD:
        raise AssertionError("Expected global dataset-max normalization")
    if ex.value_max is None or ex.value_max <= 0:
        raise AssertionError("Missing dataset raw maximum")
    if ex.scale[:2].tolist() != [1.,1.]:
        raise AssertionError("Double scaling of normalized values")
    raw_max=float(ex.df[["v0","v1"]].to_numpy().max())
    if abs(raw_max-1.)>1e-10:
        raise AssertionError(f"Normalized maximum must be one; got {raw_max}")
    validate_models(ex.models)
    return {
        "profile":PROFILE,"config_sha256":config_hash(),
        "input_sha256":ex.sha256,
        "raw_value_max":float(ex.value_max),
        "raw_value_min":float(ex.value_min),
        "normalized_value_max":raw_max,
        "max_trials":int(ex.max_trials),
    }

def ensure_equal_cross_dataset_specs(specs):
    """Used by CI/diagnostics: compare mathematical models independently of data."""
    if not specs:
        raise ValueError("No specs")
    first=specs[0]
    for s in specs[1:]:
        if s!=first:
            raise AssertionError("Cross-dataset models/priors differ")
    return True
