"""Audit neural model-evidence consistency: 4-way posterior vs 8-way conditional.

CRITICAL: feed the EXACT SAME arrays to all three trained classifiers.
Run this BEFORE any costly refits. This writes only into outputs/*/next_steps.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["ATT_BF_VALUE_NORMALIZATION"] = "dataset_max"
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from _shared import common
from experiments import normalized_value_prior_sweep as sweep

DDM = sweep.FAMILIES["ddm"]
RACE = sweep.FAMILIES["race"]
FAMILIES = {"ddm": DDM, "race": RACE}


def conditional_probabilities(eight_probs, family_names, all_names):
    arr = np.asarray(eight_probs, dtype=np.float64)
    indices = [all_names.index(m) for m in family_names]
    if arr.ndim != 2 or arr.shape[1] != len(all_names):
        raise ValueError("Wrong joint probability shape")
    joint = arr[:, indices]
    mass = joint.sum(axis=1)
    if np.any(mass <= 1e-12):
        raise ValueError("Joint family mass too small for reliable conditioning")
    return joint / mass[:, None], mass


def summarize(rows):
    if not rows:
        raise ValueError("Empty comparison")
    return {
        "n": len(rows),
        "mean_tv": float(np.mean([r["tv"] for r in rows])),
        "median_tv": float(np.median([r["tv"] for r in rows])),
        "q90_tv": float(np.quantile([r["tv"] for r in rows], .9)),
        "max_tv": float(np.max([r["tv"] for r in rows])),
        "winner_agreement": float(np.mean([r["winner_agreement"] for r in rows])),
        "mean_joint_family_mass": float(np.mean([r["joint_family_mass"] for r in rows])),
    }


def _batched_prob(estimator, raw, names, batch_size):
    k = len(names)
    n = len(raw["trials"])
    outputs = []
    for i in range(0, n, batch_size):
        current = {key: values[i:i+batch_size] for key, values in raw.items()}
        result = estimator.estimate(conditions=current,
                                    model_prior=np.full(k, 1.0 / k, dtype=float))
        p = np.asarray(result["model_probs"], dtype=np.float64)
        if p.shape != (len(current["trials"]), k):
            raise RuntimeError(f"Unexpected posterior shape {p.shape}, expected {(len(current['trials']), k)}")
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-4):
            raise RuntimeError("Nonfinite or unnormalized model probabilities")
        outputs.append(p)
    return np.concatenate(outputs, axis=0)


def _evaluate(label, ids, truth, raw, estimators, names, batch_size):
    probs = {f: _batched_prob(estimators[f], raw, names[f], batch_size)
             for f in ("ddm", "race", "all")}
    rows = []
    aggregate = {}
    for fam in ("ddm", "race"):
        conditioned, mass = conditional_probabilities(probs["all"], names[fam], names["all"])
        own = probs[fam]
        for i in range(len(ids)):
            actual_model = "" if truth is None else truth[i]
            record = {
                "dataset": label, "subject_or_draw": str(ids[i]),
                "generator": actual_model, "family": fam,
                "tv": float(.5 * np.abs(own[i] - conditioned[i]).sum()),
                "winner_agreement": int(np.argmax(own[i]) == np.argmax(conditioned[i])),
                "four_winner": names[fam][int(np.argmax(own[i]))],
                "conditioned_eight_winner": names[fam][int(np.argmax(conditioned[i]))],
                "joint_family_mass": float(mass[i]),
            }
            for j, m in enumerate(names[fam]):
                record["four__" + m] = float(own[i, j])
                record["joint_conditioned__" + m] = float(conditioned[i, j])
                record["abs_diff__" + m] = float(abs(own[i, j] - conditioned[i, j]))
            rows.append(record)
        aggregate[fam] = summarize([r for r in rows if r["family"] == fam])
        if truth is not None:
            aggregate[fam]["by_generator"] = {
                m: summarize([r for r in rows if r["family"] == fam and r["generator"] == m])
                for m in names["all"]
            }
    return rows, aggregate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=common.ROOT / "data/tavares2017/trial_eye.csv")
    p.add_argument("--mode", choices=("formal", "test"), default="formal")
    p.add_argument("--n-per-model", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--data", choices=("both", "synthetic", "real"), default="both")
    p.add_argument("--output", type=Path,
                   default=common.ROOT / "outputs/tavares2017_bf/next_steps/consistency")
    args = p.parse_args()
    if args.batch_size < 1 or args.n_per_model < 1:
        p.error("--batch-size and --n-per-model must be positive")
    common.environment(args.seed)
    sweep.configure_priors(1, 50, .5, 5, 0, 1, 0, 2)
    sweep.apply_shared_priors()
    sweep.NORMALIZATION_AUDIT = sweep.normalization_audit(args.input)
    sweep.check_checkpoints(("ddm", "race", "all"), args.mode)
    import keras
    estimators = {
        f: keras.saving.load_model(sweep.classifier_folder(f, args.mode) / "best.keras")
        for f in ("ddm", "race", "all")
    }
    names = {f: sweep.FAMILIES[f] for f in ("ddm", "race", "all")}
    ex = common.Experiment(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed, "n_per_model": args.n_per_model,
        "batch_size": args.batch_size, "input_sha256": ex.sha256,
        "classifiers": {
            f: str(sweep.classifier_folder(f, args.mode) / "best.keras")
            for f in estimators
        },
        "comparisons": {},
    }
    if args.data in ("both", "synthetic"):
        sample_blocks, truths, ids = [], [], []
        for j, model in enumerate(names["all"]):
            sim, _ = ex.simulator(common.bf, model, args.seed + 30000 + j * 1000, True)
            generated = sim.sample(args.n_per_model)
            sample_blocks.append(generated)
            truths.extend([model] * args.n_per_model)
            ids.extend([f"{model}:{i}" for i in range(args.n_per_model)])
        raw = {key: np.concatenate([block[key] for block in sample_blocks], 0)
               for key in sample_blocks[0]}
        rows, summary = _evaluate("synthetic", ids, truths, raw, estimators, names, args.batch_size)
        _write_csv(args.output / "synthetic_paired_posteriors.csv", rows)
        manifest["comparisons"]["synthetic"] = summary
        print("[SYNTHETIC]", json.dumps(summary, indent=2), flush=True)
    if args.data in ("both", "real"):
        ids, raw, audit = ex.real()
        rows, summary = _evaluate("real", ids, None, raw, estimators, names, args.batch_size)
        _write_csv(args.output / "real_paired_posteriors.csv", rows)
        _write_csv(args.output / "real_trial_audit.csv", audit)
        manifest["comparisons"]["real"] = summary
        print("[REAL]", json.dumps(summary, indent=2), flush=True)
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[SAVED]", args.output, flush=True)


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save empty CSV")
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
