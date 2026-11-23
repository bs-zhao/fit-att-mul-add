"""Isolated Food Equal Final1 aRACE_g parameter decoder repair experiment.

Train a stronger candidate (same priors/model and normalized data), compare
original vs candidate on IDENTICAL independent held-out simulator subjects,
then decode real data only if the candidate passes the diagnostic gate.

Commands:
  python race_add_decoder_repair.py train --seed 20261124
  python race_add_decoder_repair.py evaluate --seed 20261124
  python race_add_decoder_repair.py decode --seed 20261124

Candidate outputs are separate from outputs/food_equal_final1/.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["ATT_BF_VALUE_NORMALIZATION"] = "dataset_max"

HERE = Path(__file__).resolve().parent
FIT_NN = HERE.parent
REPO = FIT_NN.parents[1]
UPSTREAM = FIT_NN / "food_equal_bf"
sys.path.insert(0, str(FIT_NN))
sys.path.insert(0, str(UPSTREAM))

MODEL = "aRACE_g"
NAMES = ("d", "a", "gamma")
DEFAULT_SEED = 20261124
BASE_ROOT = REPO / "outputs" / "food_equal_final1"
REPAIR_ROOT = REPO / "outputs" / "food_equal_final1_race_add_repair"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "evaluate", "decode"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--input", type=Path, default=REPO / "data" / "food_equal" / "trial_eye.csv")
    p.add_argument("--baseline-root", type=Path, default=BASE_ROOT)
    p.add_argument("--repair-root", type=Path, default=REPAIR_ROOT)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--validation", type=int, default=4000)
    p.add_argument("--allow-unvalidated-decode", action="store_true",
                   help="Bypass diagnostic gate; not recommended")
    a = p.parse_args()
    if a.seed < 0 or any(getattr(a, x) < 1 for x in
            ("n_test", "draws", "batch_size", "epochs", "batches",
             "train_batch_size", "validation")):
        p.error("seed must be nonnegative and all count settings positive")
    if a.baseline_root.resolve() == (a.repair_root / ("seed_" + str(a.seed))).resolve():
        p.error("baseline and candidate output roots must differ")
    return a


def candidate_root(a):
    return a.repair_root / ("seed_" + str(a.seed))


def checkpoint_root(root):
    return root / MODEL / "formal"


def load_context(a):
    from unified_final1 import install, audit_experiment
    source = importlib.import_module("_shared.data")
    install(source.unified_g2)
    ex = source.Experiment(a.input, unified=True)
    audit_experiment(ex)
    from _shared import common
    if common.unified_g2 is not source.unified_g2:
        raise RuntimeError("Final1 shared module alias mismatch")
    return source, ex, common


def verified_estimator(source, ex, common, folder):
    import numpy as np
    if not (folder / "best.keras").is_file():
        raise FileNotFoundError(folder / "best.keras")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
    source.unified_g2.validate_checkpoint(
        meta, ex, [MODEL], source.FEATURE_NAMES, folder)
    if meta.get("model") != MODEL or tuple(meta.get("parameter_names", [])) != NAMES:
        raise RuntimeError("Not the expected aRACE_g parameter checkpoint")
    return common.keras.saving.load_model(folder / "best.keras")


def train(a):
    out = candidate_root(a)
    if checkpoint_root(out).joinpath("best.keras").exists():
        raise FileExistsError("Candidate checkpoint exists; choose another --seed")
    from final1_launcher import main
    main(["s1", "--dataset", "food_equal", "--model", MODEL,
          "--mode", "formal", "--seed", str(a.seed),
          "--input", str(a.input), "--output", str(out),
          "--epochs", str(a.epochs), "--batches", str(a.batches),
          "--batch-size", str(a.train_batch_size),
          "--validation", str(a.validation)])
    print("[CANDIDATE TRAINED]", checkpoint_root(out), flush=True)


def evaluate(a):
    import numpy as np
    import pandas as pd
    from _shared.diagnostics import recovery
    source, ex, common = load_context(a)
    common.environment(a.seed + 600000)
    base = checkpoint_root(a.baseline_root)
    cand = checkpoint_root(candidate_root(a))
    baseline_model = verified_estimator(source, ex, common, base)
    candidate_model = verified_estimator(source, ex, common, cand)

    dest = candidate_root(a) / "paired_evaluation"
    if (dest / "comparison.csv").exists():
        raise FileExistsError("Evaluation exists. Choose another --seed/output.")
    dest.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: generate ONE held-out data set, then use it for BOTH models.
    # Generator and priors stay at original Final1 (gamma U(0,10)).
    sim, stats = ex.simulator(common.bf, MODEL, a.seed + 700000)
    raw = sim.sample(a.n_test)
    truth = np.concatenate([raw[n] for n in NAMES], axis=-1)
    np.savez_compressed(dest / "heldout_truth.npz", truth=truth, names=NAMES)
    results = {}
    for label, estimator in (("baseline", baseline_model), ("candidate", candidate_model)):
        subdir = dest / label
        subdir.mkdir()
        samples = common.posterior(estimator, raw, NAMES, a.draws, a.batch_size)
        np.savez_compressed(subdir / "posterior.npz", samples=samples)
        recovery(samples, truth, NAMES, ex.models[MODEL]["free_pranges"], subdir)
        metrics = pd.read_csv(subdir / "parameter_recovery_metrics.csv")
        results[label] = metrics.set_index("parameter")
        print("[EVALUATED]", label, flush=True)

    table = pd.DataFrame({"parameter": list(NAMES)})
    for label in ("baseline", "candidate"):
        metrics = results[label]
        for key in ("pearson", "bias", "rmse", "coverage90", "width90", "sbc_ks", "sbc_ks_p"):
            table[label + "_" + key] = [float(metrics.loc[name, key]) for name in NAMES]
    table["abs_bias_improvement"] = table["baseline_bias"].abs() - table["candidate_bias"].abs()
    table["coverage_error_improvement"] = (
        (table["baseline_coverage90"] - .90).abs()
        - (table["candidate_coverage90"] - .90).abs())
    table["sbc_ks_improvement"] = table["baseline_sbc_ks"] - table["candidate_sbc_ks"]
    table["rmse_improvement"] = table["baseline_rmse"] - table["candidate_rmse"]
    table.to_csv(dest / "comparison.csv", index=False)

    # A conservative gate: no parameter clearly worse on coverage or RMSE;
    # original problematic a improves coverage and its SBC KS statistic.
    # This is an acceptance criterion, NOT a statistical guarantee of calibration.
    b, c = results["baseline"], results["candidate"]
    gate = bool(
        all(abs(float(c.loc[n, "coverage90"]) - .9)
            <= abs(float(b.loc[n, "coverage90"]) - .9) + .015 for n in NAMES)
        and all(float(c.loc[n, "rmse"]) <= float(b.loc[n, "rmse"]) * 1.05
                for n in NAMES)
        and abs(float(c.loc["a", "coverage90"]) - .9)
            < abs(float(b.loc["a", "coverage90"]) - .9)
        and float(c.loc["a", "sbc_ks"]) < float(b.loc["a", "sbc_ks"])
    )
    report = {
        "model": MODEL, "seed": a.seed, "heldout_seed": a.seed + 700000,
        "heldout_n": a.n_test, "posterior_draws": a.draws,
        "same_heldout_subjects": True,
        "original_model_unchanged": True,
        "generator_gamma_prior": [0.0, 10.0],
        "baseline_checkpoint": str(base / "best.keras"),
        "candidate_checkpoint": str(cand / "best.keras"),
        "gate_passed": gate,
        "gate_note": "Heuristic screening gate, not proof of posterior calibration. Inspect all SBC and coverage plots.",
        "simulation_stats": stats,
    }
    (dest / "decision.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(table.to_string(index=False), flush=True)
    print("[GATE]", "PASS" if gate else "FAIL", flush=True)
    print("[OUTPUT]", dest, flush=True)


def decode(a):
    gate_file = candidate_root(a) / "paired_evaluation" / "decision.json"
    if not a.allow_unvalidated_decode:
        if not gate_file.is_file() or not json.loads(
                gate_file.read_text(encoding="utf8")).get("gate_passed", False):
            raise RuntimeError("Candidate did not pass paired recovery gate; refusing real-data decode")
    from final1_launcher import main
    main(["s3", "--dataset", "food_equal", "--model", MODEL,
          "--mode", "formal", "--seed", str(a.seed),
          "--input", str(a.input),
          "--output", str(candidate_root(a)),
          "--draws", "2000", "--batch-size", str(a.batch_size),
          "--ppc-draws", "0"])


if __name__ == "__main__":
    args = parser()
    {"train": train, "evaluate": evaluate, "decode": decode}[args.stage](args)
