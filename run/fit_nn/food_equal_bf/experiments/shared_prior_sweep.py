"""Independent Food Equal BF experiment with SHARED d/a priors across 8 models.

Default sensitivity experiment:
    d ~ Uniform(0.1, 5.0) in ALL eight models
    a ~ Uniform(1.0, 5.0) in ALL eight models
    Mul theta ~ Uniform(0.0, 1.0), Add gamma ~ Uniform(0.0, 5.0)
Command-line --d-min/max and --a-min/max allow further PRE-SPECIFIED
shared prior ranges. Each distinct configuration gets its own output tree.
NoAtt/AttOnly and all other parameters are unchanged.

The original food_equal model_info.py and food_equal_bf default S1-S6
are NEVER changed. S4/S5/S6 and cross-architecture diagnostics reuse the
existing implementations, but all artifacts are isolated under OUTPUT.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _shared.common import (ROOT, MODEL_NAMES, model_infos, parser, training_args,
                            train, directory, dump, positive, nonnegative)
from _shared.data import prior_spec, ATTENTION_PRIORS, PRIOR_PROFILE
from experiments.wide_prior_classifier import verify_or_write_spec

D_PRIOR = (0.1, 5.0)
A_PRIOR = (1.0, 5.0)
EXPERIMENT_PREFIX = "sweep_v1_shared"


def _prior_tag(value):
    """Unambiguous stable directory token for practical decimal prior endpoints."""
    value = float(value)
    return str(value).replace("-", "m").replace(".", "p").replace("+", "")


def experiment_id(d, a):
    return (f"{EXPERIMENT_PREFIX}_d{_prior_tag(d[0])}to{_prior_tag(d[1])}"
            f"_a{_prior_tag(a[0])}to{_prior_tag(a[1])}"
            "_theta0to1_gamma0to5")


EXPERIMENT_ID = experiment_id(D_PRIOR, A_PRIOR)
OUTPUT = ROOT / "outputs/food_equal_bf/prior_sensitivity" / EXPERIMENT_ID


def configure_priors(d_min, d_max, a_min, a_max):
    """Set one homogeneous prior for each parameter, for all eight models.

    No implicit output reuse: the directory name encodes the full configuration,
    and the sweep_v1 prefix is distinct from EVERY preceding experiment.
    """
    global D_PRIOR, A_PRIOR, EXPERIMENT_ID, OUTPUT
    bounds = [float(d_min), float(d_max), float(a_min), float(a_max)]
    if not all(math.isfinite(x) for x in bounds):
        raise ValueError("All prior endpoints must be finite")
    if not (0 < bounds[0] < bounds[1] and 0 < bounds[2] < bounds[3]):
        raise ValueError("Require 0 < d-min < d-max and 0 < a-min < a-max")
    d, a = tuple(bounds[:2]), tuple(bounds[2:])
    D_PRIOR, A_PRIOR = d, a
    EXPERIMENT_ID = experiment_id(d, a)
    OUTPUT = ROOT / "outputs/food_equal_bf/prior_sensitivity" / EXPERIMENT_ID
FAMILIES = {
    "race": ["aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g"],
    "ddm": ["aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g"],
    "all": list(MODEL_NAMES),
}


def apply_shared_da_priors():
    """Mutate only this process's independent BF model_infos copy.

    Ordinary S4/S5/S6 entry points launched in other processes retain their
    original d/a ranges. Do not edit the legacy model_info.py on disk.
    """
    for name, info in model_infos.items():
        for name_key, range_key in (("free_pnames", "free_pranges"),
                                    ("free_pnames_ml", "free_pranges_ml")):
            bounds = list(info[range_key])
            for param, interval in (("d", D_PRIOR), ("a", A_PRIOR)):
                i = info[name_key].index(param)
                bounds[i] = interval
            info[range_key] = bounds
    validate_priors()


def validate_priors():
    """Fail fast if any model deviates from the requested shared ranges."""
    for name, info in model_infos.items():
        for name_key, range_key in (("free_pnames", "free_pranges"),
                                    ("free_pnames_ml", "free_pranges_ml")):
            bounds = dict(zip(info[name_key], info[range_key]))
            if bounds["d"] != D_PRIOR or bounds["a"] != A_PRIOR:
                raise AssertionError(f"{name}: inconsistent d/a priors: {bounds}")
            if name.endswith("_t") and bounds["theta"] != ATTENTION_PRIORS["theta"]:
                raise AssertionError(f"{name}: theta prior is not wide")
            if name.endswith("_g") and bounds["gamma"] != ATTENTION_PRIORS["gamma"]:
                raise AssertionError(f"{name}: gamma prior is not wide")


def expected_spec(names, mode):
    validate_priors()
    return {
        "experiment": EXPERIMENT_ID,
        "mode": mode,
        "model_names": list(names),
        "d": list(D_PRIOR),
        "a": list(A_PRIOR),
        "theta": list(ATTENTION_PRIORS["theta"]),
        "gamma": list(ATTENTION_PRIORS["gamma"]),
        "default_attention_profile": PRIOR_PROFILE,
        "free_priors": prior_spec(names)["free_priors"],
    }


def classifier_folder(family, mode):
    names = FAMILIES[family]
    args = parser("shared d/a checkpoint", classifier=True).parse_args([
        "--mode", mode, "--output", str(OUTPUT), "--models", *names
    ])
    return directory(args, classifier=True)


def check_checkpoints(families, mode):
    for family in families:
        folder = classifier_folder(family, mode)
        verify_or_write_spec(folder, expected_spec(FAMILIES[family], mode), "s5")
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
        # _shared.common.load also checks its prior_spec against in-memory
        # free priors before deserializing the network.
        if metadata.get("prior_spec") != prior_spec(FAMILIES[family]):
            raise ValueError(f"Checkpoint was trained with different priors: {folder}")
        if metadata.get("model_names") != FAMILIES[family]:
            raise ValueError(f"Unexpected model set in checkpoint: {folder}")
        print("[VERIFIED]", family, folder / "best.keras", flush=True)


def cli_for_stage(args, seed):
    names = FAMILIES[args.family]
    cli = ["--mode", args.mode, "--seed", str(seed),
           "--input", str(args.input), "--output", str(OUTPUT),
           "--models", *names]
    if args.stage == "s4":
        for flag, value in (("--epochs", args.epochs), ("--batches", args.batches),
                            ("--batch-size", args.batch_size),
                            ("--validation", args.validation)):
            if value is not None:
                cli.extend((flag, str(value)))
        if args.overwrite:
            cli.append("--overwrite")
    if args.stage == "s5":
        for flag, value in (("--n-per-model", args.n_per_model),
                            ("--batch-size", args.batch_size)):
            if value is not None:
                cli.extend((flag, str(value)))
    return cli


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["s4", "s5", "s6", "cross", "report"],
                   required=True)
    p.add_argument("--family", choices=["race", "ddm", "all", "both"],
                   help="s4/s5/s6: race, ddm, all; report: race, ddm, both")
    p.add_argument("--mode", choices=["test", "formal"], default="formal")
    p.add_argument("--d-min", type=float, default=0.1)
    p.add_argument("--d-max", type=float, default=5.0)
    p.add_argument("--a-min", type=float, default=1.0)
    p.add_argument("--a-max", type=float, default=5.0)
    p.add_argument("--input", type=Path, default=ROOT / "data/food_equal/trial_eye.csv")
    p.add_argument("--seed", type=nonnegative)
    p.add_argument("--epochs", type=positive)
    p.add_argument("--batches", type=positive)
    p.add_argument("--batch-size", type=positive)
    p.add_argument("--validation", type=positive)
    p.add_argument("--n-per-model", type=positive)
    p.add_argument("--n", type=positive, help="Cross-architecture accepted simulated participants")
    p.add_argument("--bootstrap", type=positive)
    p.add_argument("--generator", choices=["aRACE_g", "aRACE_t"],
                   default="aRACE_g", help="Cross-architecture generating model")
    p.add_argument("--overwrite", action="store_true",
                   help="Explicit S4 retraining within the NEW experiment only")
    a = p.parse_args()
    try:
        configure_priors(a.d_min, a.d_max, a.a_min, a.a_max)
    except ValueError as err:
        p.error(str(err))

    if a.stage in ("s4", "s5", "s6"):
        if a.family not in FAMILIES:
            p.error("S4/S5/S6 require --family race, ddm, or all")
        if a.n is not None or a.bootstrap is not None:
            p.error("--n and --bootstrap are cross-only options")
        if a.stage != "s4" and any(v is not None for v in
                                   (a.epochs, a.batches, a.validation)):
            p.error("--epochs/--batches/--validation are S4-only")
        if a.stage != "s4" and a.overwrite:
            p.error("--overwrite is S4-only")
        if a.stage != "s5" and a.n_per_model is not None:
            p.error("--n-per-model is S5-only")
        if a.stage == "s6" and a.batch_size is not None:
            p.error("--batch-size is not supported by S6")
    elif a.stage == "cross":
        if a.family is not None:
            p.error("Cross-architecture comparison uses both four-model classifiers; omit --family")
        if any(v is not None for v in
               (a.epochs, a.batches, a.validation, a.n_per_model)) or a.overwrite:
            p.error("S4/S5-only options are invalid for --stage cross")
    else:
        if a.family not in ("race", "ddm", "both"):
            p.error("Report requires --family race, ddm, or both")
        if any(v is not None for v in
               (a.epochs, a.batches, a.batch_size, a.validation, a.n_per_model,
                a.n, a.bootstrap)) or a.overwrite:
            p.error("Report uses only --stage report, --family, and --mode")

    apply_shared_da_priors()
    seed = a.seed if a.seed is not None else (20261102 if a.stage == "cross" else 20261030)
    print(f"[EXPERIMENT] {EXPERIMENT_ID}", flush=True)
    print(f"[PRIORS] d={D_PRIOR} a={A_PRIOR} theta={ATTENTION_PRIORS['theta']} "
          f"gamma={ATTENTION_PRIORS['gamma']}", flush=True)
    print(f"[OUTPUT ROOT] {OUTPUT}", flush=True)

    if a.stage == "report":
        from experiments.report_model_comparison import generate
        for family in (("race", "ddm") if a.family == "both" else (a.family,)):
            check_checkpoints((family,), a.mode)
        generate(OUTPUT, a.mode, a.family)
        return

    if a.stage == "cross":
        check_checkpoints(("ddm", "race"), a.mode)
        argv = [
            "race_add_cross_architecture.py",
            "--generator", a.generator,
            "--mode", a.mode,
            "--input", str(a.input),
            "--output", str(OUTPUT),
            "--seed", str(seed),
            "--n", str(a.n or (12 if a.mode == "test" else 1000)),
            "--batch-size", str(a.batch_size or 25),
            "--bootstrap", str(a.bootstrap or 5000),
        ]
        # The shared d/a override is in process before invoking the existing
        # paired four-DDM vs four-RACE cross-architecture analysis.
        from experiments.race_add_cross_architecture import main as compare
        sys.argv = argv
        compare(default_generator=a.generator)
        return

    names = FAMILIES[a.family]
    folder = classifier_folder(a.family, a.mode)
    if a.stage == "s4":
        # Prior provenance file uses a unique experiment ID and full bounds.
        verify_or_write_spec(folder, expected_spec(names, a.mode), "s4")
    else:
        check_checkpoints((a.family,), a.mode)
    cli = cli_for_stage(a, seed)
    if a.stage == "s4":
        options = training_args(parser("Shared d/a S4", classifier=True)).parse_args(cli)
        train(options, classifier=True)
    else:
        sys.argv = [f"{a.stage}_model_comparison.py", *cli]
        if a.stage == "s5":
            from s5_model_recovery import main as evaluate
        else:
            from s6_classify_real_data import main as evaluate
        evaluate()
    print(f"[DONE] {a.stage} {a.family} {folder}", flush=True)


if __name__ == "__main__":
    main()
