"""Final2 S3-fitted RACE-Add -> independently trained DDM4 cross-architecture test.

For every real subject, fix posterior MEAN (d, a, gamma) estimated in S3;
simulate ten fresh datasets using that subject's actual trial designs;
classify each using the existing S4 DDM4 best.keras (uniform four-model prior).
No new training and no re-sampling parameter priors.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from unified_final2 import DATASETS, PRIORS, PROFILE, install, audit_experiment

DDM = ["aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g"]
LABELS = ["NoAtt", "AttOnly", "Mul", "Add"]
GENERATOR = "aRACE_g"
REQUIRED_PARAMS = ("d", "a", "gamma")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--mode", choices=("test", "formal"), default="formal")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=10, help="DDM inference batch size")
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.repeats < 1 or a.batch_size < 1 or a.seed < 0:
        p.error("--repeats and --batch-size must be positive; seed must be nonnegative")
    return a


def validate_s3(np, file, ids):
    with np.load(file, allow_pickle=False) as stored:
        if not all(k in stored for k in ("samples", "subject_ids", "names")):
            raise ValueError("S3 posterior NPZ missing samples/subject_ids/names")
        samples = np.asarray(stored["samples"], dtype=float)
        subject_ids = [str(x) for x in stored["subject_ids"].tolist()]
        names = [str(x) for x in stored["names"].tolist()]
    if len(names) != len(set(names)) or set(names) != set(REQUIRED_PARAMS):
        raise ValueError(f"S3 expected parameter names {REQUIRED_PARAMS}, got {names}")
    if samples.ndim != 3 or samples.shape[0] != len(subject_ids) or samples.shape[2] != len(names):
        raise ValueError(f"Invalid S3 posterior dimensions {samples.shape}")
    if samples.shape[1] == 0 or not np.isfinite(samples).all():
        raise ValueError("Empty or nonfinite S3 posterior")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Duplicate S3 subject IDs")
    real_ids = [str(x) for x in ids]
    if len(set(real_ids)) != len(real_ids) or set(real_ids) != set(subject_ids):
        raise ValueError("S3 subject IDs differ from current real-data subjects")
    ix = {sid: j for j, sid in enumerate(subject_ids)}
    means = samples.mean(axis=1)
    selected = {sid: dict(zip(names, means[ix[sid]].tolist())) for sid in real_ids}
    for sid, par in selected.items():
        for name, value in par.items():
            lo, hi = PRIORS[name]
            if not (lo - 1e-5 <= value <= hi + 1e-5):
                raise ValueError(f"{sid} posterior mean {name}={value} outside Final2 prior [{lo}, {hi}]")
    return selected, samples.shape[1]


def run(argv=None):
    a = parse(argv)
    import numpy as np

    data = a.input or ROOT / "data" / a.dataset / "trial_eye.csv"
    output = a.output or ROOT / "outputs" / f"{a.dataset}_final2"
    upstream = HERE / f"{a.dataset}_bf"
    sys.path.insert(0, str(upstream))
    source = importlib.import_module("_shared.data")
    install(source.unified_g2)
    ex = source.Experiment(data, unified=True)
    audit = audit_experiment(ex)
    ids, real, trial_audit = ex.real()
    posterior_dir = output / GENERATOR / a.mode
    s3_file = posterior_dir / "real_data" / "posterior_samples.npz"
    ddm_dir = output / "classifier" / "__".join(DDM) / a.mode
    s1_meta = json.loads((posterior_dir / "metadata.json").read_text(encoding="utf8"))
    ddm_meta = json.loads((ddm_dir / "metadata.json").read_text(encoding="utf8"))
    source.unified_g2.validate_checkpoint(s1_meta, ex, [GENERATOR], source.FEATURE_NAMES, posterior_dir)
    source.unified_g2.validate_checkpoint(ddm_meta, ex, DDM, source.FEATURE_NAMES, ddm_dir)
    if s1_meta.get("model") != GENERATOR or s1_meta.get("parameter_names") != list(REQUIRED_PARAMS):
        raise ValueError("S1 parameter-decoder metadata mismatch")
    if ddm_meta.get("model_names") != DDM or not np.allclose(ddm_meta.get("model_prior"), [0.25]*4):
        raise ValueError("DDM4 metadata model names/prior mismatch")
    if not (ddm_dir / "best.keras").is_file():
        raise FileNotFoundError(f"Missing S4 DDM4 classifier: {ddm_dir / 'best.keras'}")
    selected, n_draws = validate_s3(np, s3_file, ids)

    dest = (output / "cross_architecture" /
            "S3_fitted_aRACE_g__tested_DDM4" / a.mode /
            f"seed_{a.seed}_repeats_{a.repeats}")
    print("[FINAL1 S3-FITTED CROSS PREFLIGHT]", json.dumps({
        "dataset": a.dataset, "profile": PROFILE, "normalization": audit,
        "n_subjects": len(ids), "s3_posterior_draws_per_subject": n_draws,
        "point_estimate": "posterior mean (d,a,gamma)",
        "repeat_simulations_per_subject": a.repeats,
        "simulation_design": "each subject's own empirical values and gaze schedules",
        "s3_file": str(s3_file), "ddm4_checkpoint": str(ddm_dir / "best.keras"),
        "destination": str(dest)
    }, indent=2), flush=True)
    if a.dry_run:
        return
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {dest}")

    from _shared.common import environment, keras, dump
    from _shared.data import funcs, pack
    from experiments.race_add_cross_architecture import classify
    import pandas as pd

    environment(a.seed)
    estimator = keras.saving.load_model(ddm_dir / "best.keras")
    # S3 real-data ID alignment and subject-specific trial design are guaranteed.
    group_map = {str(sid): group for sid, group in ex.groups}
    rng = np.random.default_rng(a.seed + 800000)
    counters = dict(trials=0, nonhits=0, maxT_excluded=0, rejected_subjects=0, accepted_subjects=0)
    rows = []
    pending = []
    meta = []

    def flush():
        if not pending:
            return
        cond = {key: np.stack([rec[key] for rec in pending])
                for key in ("trials", "summary_mask", "n_trials")}
        prob, bf = classify(estimator, cond, DDM)
        for i, info in enumerate(meta):
            item = dict(info)
            item["DDM_logBF_Mul_Add"] = float(bf[i])
            item["DDM_best_model"] = DDM[int(prob[i].argmax())]
            item.update({f"DDM_p_{label}": float(prob[i, j]) for j, label in enumerate(LABELS)})
            rows.append(item)
        pending.clear()
        meta.clear()

    # Deterministic subject order is inherited from ex.real() and cross-checked against S3 IDs.
    for sidx, sid in enumerate(ids):
        key = str(sid)
        group = group_map[key]
        fitted = selected[key]
        for rep in range(a.repeats):
            # Fixed-parameter dictionary: no new random draws from the priors.
            params = dict(zip(ex.models[GENERATOR]["fixed_pnames"],
                              ex.models[GENERATOR]["fixed_pvalues"]))
            params.update(fitted)
            _, sim_rows = ex.simulate(GENERATOR, rng, counters, params=params, group=group)
            if len(sim_rows) <= 5:
                counters["rejected_subjects"] += 1
                # Record attempted replication; do NOT silently replace failures.
                rows.append(dict(subject_id=key, subject_index=sidx, repeat=rep,
                                 true_d=fitted["d"], true_a=fitted["a"],
                                 true_gamma=fitted["gamma"], n_retained_trials=len(sim_rows),
                                 status="insufficient_retained_trials",
                                 DDM_logBF_Mul_Add=np.nan, DDM_best_model="",
                                 **{f"DDM_p_{label}": np.nan for label in LABELS}))
                continue
            counters["accepted_subjects"] += 1
            pending.append(pack(sim_rows, ex.max_trials))
            meta.append(dict(subject_id=key, subject_index=sidx, repeat=rep,
                             true_d=fitted["d"], true_a=fitted["a"],
                             true_gamma=fitted["gamma"], n_retained_trials=len(sim_rows),
                             status="ok"))
            if len(pending) >= a.batch_size:
                flush()
        print(f"[CROSS] completed subject {sidx + 1}/{len(ids)}: {key}", flush=True)
    flush()
    frame = pd.DataFrame(rows).sort_values(["subject_index", "repeat"]).reset_index(drop=True)
    if len(frame) != len(ids)*a.repeats:
        raise AssertionError("Missing subject-repetition output rows")
    valid = frame.loc[frame.status == "ok"]
    if len(valid) == 0:
        raise RuntimeError("No valid simulated datasets")
    probs = {name: float(valid[f"DDM_p_{label}"].mean()) for name, label in zip(DDM, LABELS)}
    winners = valid.DDM_best_model.value_counts()
    logbf = valid.DDM_logBF_Mul_Add.to_numpy()
    subject = (valid.groupby("subject_id", sort=False)
               .agg(n_valid=("repeat", "count"),
                    mean_logBF_Mul_Add=("DDM_logBF_Mul_Add", "mean"),
                    mul_wins_over_add=("DDM_logBF_Mul_Add", lambda x: int((x > 0).sum())),
                    ddm_add_wins=("DDM_best_model", lambda x: int((x == "aDDM_g").sum())),
                    ddm_mul_wins=("DDM_best_model", lambda x: int((x == "aDDM_t").sum())),
                    mean_p_mul=("DDM_p_Mul", "mean"),
                    mean_p_add=("DDM_p_Add", "mean"))
               .reset_index())
    for name in REQUIRED_PARAMS:
        subject[name] = subject["subject_id"].map({sid: pars[name] for sid, pars in selected.items()})
    summary = {
        "experiment": f"{a.dataset}_final2_s3_fitted_race_add_to_ddm4",
        "profile": PROFILE, "generator": GENERATOR,
        "parameter_source": "S3 posterior mean per real participant",
        "s3_samples_file": str(s3_file), "s3_posterior_draws_per_subject": n_draws,
        "subject_design": "subject-specific empirical option values and gaze fixations",
        "n_real_subjects": len(ids), "repeats_per_subject": a.repeats,
        "n_attempted": len(frame), "n_valid": len(valid), "n_invalid": len(frame)-len(valid),
        "DDM_winner_counts": {name: int(winners.get(name, 0)) for name in DDM},
        "DDM_mean_probabilities": probs,
        "mean_logBF_Mul_Add": float(logbf.mean()),
        "median_logBF_Mul_Add": float(np.median(logbf)),
        "Mul_wins_over_Add": int((logbf > 0).sum()),
        "Add_wins_over_Mul": int((logbf < 0).sum()),
        "subjects_mean_logBF_Mul_gt_Add": int((subject.mean_logBF_Mul_Add > 0).sum()),
        "subjects_mean_logBF_Add_gt_Mul": int((subject.mean_logBF_Mul_Add < 0).sum()),
        "checkpoint": str(ddm_dir / "best.keras"),
        "input_sha256": ex.sha256, "source_value_max": ex.value_max,
        "simulation_stats": counters,
        "seed": a.seed
    }
    dest.mkdir(parents=True, exist_ok=False)
    frame.to_csv(dest / "per_simulation.csv", index=False)
    subject.to_csv(dest / "per_subject.csv", index=False)
    dump(dest / "summary.json", summary)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(logbf, bins=40)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(xlabel="DDM log BF (Mul / Add)", ylabel="Simulated datasets",
           title=f"{a.dataset}: S3-fitted RACE-Add to DDM4")
    fig.tight_layout()
    fig.savefig(dest / "ddm_logBF_mul_add.png", dpi=160)
    plt.close(fig)
    print("[CROSS RESULT]", json.dumps(summary, indent=2), flush=True)
    print("[OUTPUT]", dest, flush=True)


if __name__ == "__main__":
    run()
