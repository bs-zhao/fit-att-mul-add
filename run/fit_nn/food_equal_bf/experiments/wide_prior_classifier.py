"""Optional food_equal prior sensitivity (NOT the default S1-S6 priors): gamma U(0,5), theta U(0,1).

Stages s4, s5, s6 call the unchanged BayesFlow training/recovery/real-data
entry points. New and original artifacts are kept in separate directories.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared.common import (ROOT, model_infos, parser, training_args, train,
                            directory, dump, positive, nonnegative)

FAMILIES = {
    "race": ["aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g"],
    "ddm": ["aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g"],
}
OUTPUT = ROOT / "outputs/food_equal_bf/prior_sensitivity/gamma_0_5_theta_0_1"
GAMMA_PRIOR = (0.0, 5.0)
THETA_PRIOR = (0.0, 1.0)


def apply_wide_priors():
    """Change only Mul theta and Add gamma priors in this process."""
    for names in FAMILIES.values():
        for name, parameter, interval in (
            (names[2], "theta", THETA_PRIOR),
            (names[3], "gamma", GAMMA_PRIOR),
        ):
            info = model_infos[name]
            index = info["free_pnames"].index(parameter)
            ranges = list(info["free_pranges"])
            ranges[index] = interval
            info["free_pranges"] = ranges
            if parameter in info.get("free_pnames_ml", []):
                index_ml = info["free_pnames_ml"].index(parameter)
                ml_ranges = list(info["free_pranges_ml"])
                ml_ranges[index_ml] = interval
                info["free_pranges_ml"] = ml_ranges


def expected_spec(names, mode):
    return {
        "prior_experiment": "gamma_0_5_theta_0_1",
        "mode": mode,
        "model_names": names,
        "gamma": list(GAMMA_PRIOR),
        "theta": list(THETA_PRIOR),
        "free_priors": {
            name: {p: list(map(float, bounds)) for p, bounds in zip(
                model_infos[name]["free_pnames"], model_infos[name]["free_pranges"])}
            for name in names
        },
    }


def verify_or_write_spec(folder, spec, stage):
    path = folder / "prior_spec.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf8")) != spec:
            raise ValueError(f"Prior mismatch: {path}")
    elif stage == "s4":
        if (folder / "best.keras").exists():
            raise FileExistsError(f"Unverified existing checkpoint: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        dump(path, spec)
    else:
        raise FileNotFoundError(f"Run S4 first (missing {path})")
    if stage != "s4":
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
        if metadata["model_names"] != spec["model_names"]:
            raise ValueError(f"Checkpoint model mismatch: {folder}")
        if not (folder / "best.keras").exists():
            raise FileNotFoundError(f"Missing trained classifier: {folder / 'best.keras'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["s4", "s5", "s6", "report"], required=True)
    p.add_argument("--family", choices=["race", "ddm", "both"], required=True)
    p.add_argument("--mode", choices=["formal", "test"], default="formal")
    p.add_argument("--seed", type=nonnegative, default=20261030)
    p.add_argument("--input", type=Path, default=ROOT / "data/food_equal/trial_eye.csv")
    p.add_argument("--epochs", type=positive)
    p.add_argument("--batches", type=positive)
    p.add_argument("--batch-size", type=positive)
    p.add_argument("--validation", type=positive)
    p.add_argument("--n-per-model", type=positive)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.stage == "report":
        from experiments.report_model_comparison import generate
        if a.epochs or a.batches or a.batch_size or a.validation or a.n_per_model or a.overwrite:
            p.error("Report stage accepts only --family and --mode")
        generate(OUTPUT, a.mode, a.family)
        return
    if a.family == "both":
        p.error("--family both is only valid with --stage report")
    if a.stage != "s4" and (a.epochs or a.batches or a.validation or a.overwrite):
        p.error("S4-only arguments on S5/S6")
    if a.stage != "s5" and a.n_per_model:
        p.error("--n-per-model is S5-only")
    if a.stage == "s6" and a.batch_size:
        p.error("--batch-size is not accepted by S6")

    apply_wide_priors()
    names = FAMILIES[a.family]
    cli = ["--mode", a.mode, "--seed", str(a.seed),
           "--input", str(a.input), "--output", str(OUTPUT),
           "--models", *names]
    if a.stage == "s4":
        for flag, value in (("--epochs", a.epochs), ("--batches", a.batches),
                            ("--batch-size", a.batch_size),
                            ("--validation", a.validation)):
            if value is not None:
                cli.extend([flag, str(value)])
        if a.overwrite:
            cli.append("--overwrite")
    elif a.stage == "s5":
        for flag, value in (("--n-per-model", a.n_per_model),
                            ("--batch-size", a.batch_size)):
            if value is not None:
                cli.extend([flag, str(value)])

    path_args = parser("checkpoint path", classifier=True).parse_args(
        ["--mode", a.mode, "--output", str(OUTPUT), "--models", *names]
    )
    folder = directory(path_args, classifier=True)
    verify_or_write_spec(folder, expected_spec(names, a.mode), a.stage)
    print("[PRIORS] gamma=(0,5), theta=(0,1)", flush=True)
    print("[MODELS]", names, flush=True)
    print("[PATH]", folder, flush=True)

    if a.stage == "s4":
        fit_args = training_args(parser("wide prior S4", classifier=True)).parse_args(cli)
        train(fit_args, classifier=True)
    else:
        # Reuse original S5/S6 entrypoints without editing their code.
        sys.argv = [f"{a.stage}_model_comparison.py", *cli]
        if a.stage == "s5":
            from s5_model_recovery import main as run_stage
        else:
            from s6_classify_real_data import main as run_stage
        run_stage()
        # S5/S6 report generation uses saved CSVs and never reruns inference.
        # It starts only once both S5 and S6 have completed.
        from experiments.report_model_comparison import generate
        def ready(check_folder):
            return all((check_folder / p).is_file() for p in (
                "diagnostics/predictions.csv",
                "diagnostics/reliability.csv",
                "diagnostics/classwise_reliability.csv",
                "diagnostics/metrics.json",
                "real_data/model_probabilities.csv",
                "real_data/pairwise_log_bayes_factors.csv"))
        if ready(folder):
            other = "ddm" if a.family == "race" else "race"
            other_folder = OUTPUT / "classifier" / "__".join(FAMILIES[other]) / a.mode
            generate(OUTPUT, a.mode, "both" if ready(other_folder) else a.family)
        else:
            print("[REPORT] Awaiting both S5 and S6 outputs", flush=True)
    print("[DONE]", a.stage, a.family, folder, flush=True)


if __name__ == "__main__":
    main()
