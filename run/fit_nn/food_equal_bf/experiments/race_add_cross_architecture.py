"""Paired four-DDM vs four-RACE inference on RACE-Add or RACE-Mul simulated data.

Uses existing S4 best.keras checkpoints. Does not train or alter classifiers.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats
from _shared.common import ROOT, bf, load, dump, positive, nonnegative, model_infos

DDM = ["aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g"]
RACE = ["aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g"]
GENERATOR = "aRACE_g"  # Backward-compatible default
GENERATORS = ("aRACE_g", "aRACE_t")
INPUT_KEYS = ("trials", "summary_mask", "n_trials")
LABELS = ["NoAtt", "AttOnly", "Mul", "Add"]


def classify(estimator, conditions, names):
    """Use model log evidence, avoiding unstable log(PMP_Mul/PMP_Add)."""
    result = estimator.estimate(conditions=conditions, model_prior=np.full(4, 0.25))
    probs = np.asarray(result["model_probs"], dtype=float)
    evidence = np.asarray(result["log_bayes_factors"], dtype=float)
    n = len(conditions["trials"])
    if probs.shape != (n, 4) or evidence.shape != (n, 4):
        raise ValueError(f"Expected ({n}, 4): got {probs.shape}, {evidence.shape}")
    if not np.isfinite(probs).all() or not np.isfinite(evidence).all():
        raise ValueError("Non-finite model outputs")
    if (probs < -1e-6).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Invalid posterior model probabilities")
    z = evidence - evidence.max(axis=1, keepdims=True)
    expected = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    if not np.allclose(probs, expected, atol=1e-3):
        raise ValueError("Model probabilities do not agree with log Bayes factors")
    return probs, evidence[:, 2] - evidence[:, 3]  # Canonical Mul minus Add log BF


def paired_summary(frame, n_bootstrap, seed, generator=GENERATOR):
    """Within-simulated-participant contrast of two log BF(Mul/Add) estimates."""
    ddm = frame["DDM_logBF_Mul_Add"].to_numpy(dtype=float)
    race = frame["RACE_logBF_Mul_Add"].to_numpy(dtype=float)
    delta = ddm - race
    n = len(delta)
    if n < 2:
        raise ValueError("At least two participants required")
    test = stats.ttest_rel(ddm, race)
    one_sided = stats.ttest_rel(ddm, race, alternative="greater")
    t_ci = stats.t.interval(0.95, n-1, loc=delta.mean(), scale=stats.sem(delta))
    rng = np.random.default_rng(seed)
    bootstrap = np.array([delta[rng.integers(0, n, n)].mean()
                          for _ in range(n_bootstrap)])
    try:
        wilcoxon_p = float(stats.wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.
    except ValueError:
        wilcoxon_p = None

    def describe(bf):
        return {
            "mean_logBF_Mul_Add": float(bf.mean()),
            "median_logBF_Mul_Add": float(np.median(bf)),
            "Mul_wins_vs_Add": int((bf > 0).sum()),
            "Add_wins_vs_Mul": int((bf < 0).sum()),
            "strong_Mul_BF_gt_10": int((bf > np.log(10)).sum()),
            "strong_Add_BF_gt_10": int((bf < -np.log(10)).sum()),
        }
    return {
        "generator": generator, "n": n, "DDM": describe(ddm), "RACE": describe(race),
        "delta_definition": "DDM log BF(Mul/Add) - RACE log BF(Mul/Add)",
        "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
        "sd_delta": float(delta.std(ddof=1)),
        "n_DDM_stronger": int((delta > 0).sum()),
        "n_RACE_stronger": int((delta < 0).sum()),
        "paired_t": float(test.statistic), "paired_t_p_two_sided": float(test.pvalue),
        "paired_t_p_one_sided_DDM_stronger": float(one_sided.pvalue),
        "paired_wilcoxon_p_two_sided": wilcoxon_p,
        "mean_delta_t_ci95": [float(x) for x in t_ci],
        "mean_delta_bootstrap_ci95": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
        "n_bootstrap": n_bootstrap,
    }


def plots(df, out, generator=GENERATOR):
    import matplotlib.pyplot as plt
    x = df["RACE_logBF_Mul_Add"].to_numpy()
    y = df["DDM_logBF_Mul_Add"].to_numpy()
    delta = y - x
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=14, alpha=.4)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Equal log BF")
    ax.axhline(0, color="gray", lw=.8)
    ax.axvline(0, color="gray", lw=.8)
    ax.set(xlabel="RACE log BF(Mul/Add)", ylabel="DDM log BF(Mul/Add)",
           title=f"Paired predictions on identical {generator}-generated data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "paired_logBF_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(delta, bins=min(40, max(10, int(np.sqrt(len(delta))))), edgecolor="white")
    ax.axvline(0, ls="--", color="black")
    ax.axvline(delta.mean(), color="tab:red", label="Mean")
    ax.set(xlabel="DDM minus RACE log BF(Mul/Add)", ylabel="Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "paired_delta_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    attention_name = "gamma" if generator == "aRACE_g" else "theta"
    ax.scatter(df[f"true_{attention_name}"], delta, s=14, alpha=.35)
    ax.axhline(0, ls="--", color="black")
    ax.set(xlabel=f"True {generator} {attention_name}", ylabel="DDM minus RACE log BF(Mul/Add)")
    fig.tight_layout()
    fig.savefig(out / f"paired_delta_vs_{attention_name}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ix = np.arange(4)
    ax.bar(ix-.2, [df[f"DDM_p_{name}"].mean() for name in LABELS], width=.4,
           label="DDM classifier")
    ax.bar(ix+.2, [df[f"RACE_p_{name}"].mean() for name in LABELS], width=.4,
           label="RACE classifier")
    ax.set(xticks=ix, xticklabels=LABELS, ylim=(0, 1),
           ylabel="Mean posterior model probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mean_model_probs.png", dpi=180)
    plt.close(fig)


def main(default_generator=GENERATOR):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", choices=GENERATORS, default=default_generator,
                   help="Use aRACE_g (Add) or aRACE_t (Mul) to generate held-out data")
    p.add_argument("--mode", choices=["test", "formal"], default="formal")
    p.add_argument("--input", type=Path, default=ROOT / "data/food_equal/trial_eye.csv")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/food_equal_bf")
    p.add_argument("--seed", type=nonnegative, default=20261102)
    p.add_argument("--n", type=positive, help="Accepted simulated subjects (formal default: 1000)")
    p.add_argument("--batch-size", type=positive, default=25)
    p.add_argument("--bootstrap", type=positive, default=5000)
    a = p.parse_args()
    n = a.n or (12 if a.mode == "test" else 1000)

    def classifier_args(names):
        return SimpleNamespace(input=a.input, output=a.output, seed=a.seed,
                               mode=a.mode, models=names)

    print("[LOAD] 4-DDM checkpoint", flush=True)
    ddm_est, ddm_ex, ddm_folder = load(classifier_args(DDM), True)
    print("[LOAD] 4-RACE checkpoint", flush=True)
    race_est, race_ex, race_folder = load(classifier_args(RACE), True)
    if ddm_ex.sha256 != race_ex.sha256 or ddm_ex.max_trials != race_ex.max_trials:
        raise RuntimeError("DDM/RACE training data designs differ")
    for names, folder in [(DDM, ddm_folder), (RACE, race_folder)]:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
        if meta["model_names"] != names or not np.allclose(meta["model_prior"], [.25]*4):
            raise RuntimeError(f"Unexpected model order or training prior in {folder}")

    # Generate only after both loads (load reseeds the simulator's global C RNG).
    sim, counters = race_ex.simulator(bf, a.generator, a.seed + 800000,
                                      model_only=False)
    attention_name = "gamma" if a.generator == "aRACE_g" else "theta"
    out = a.output / "cross_architecture" / f"generated_{a.generator}" / a.mode / f"seed_{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for start in range(0, n, a.batch_size):
        k = min(a.batch_size, n - start)
        raw = sim.sample(k)
        conditions = {key: raw[key] for key in INPUT_KEYS}
        ddm_p, ddm_bf = classify(ddm_est, conditions, DDM)
        race_p, race_bf = classify(race_est, conditions, RACE)
        row = pd.DataFrame({
            "sim_id": np.arange(start, start+k),
            "generator": a.generator,
            "true_d": np.asarray(raw["d"]).reshape(-1),
            "true_a": np.asarray(raw["a"]).reshape(-1),
            f"true_{attention_name}": np.asarray(raw[attention_name]).reshape(-1),
            "n_retained_trials": np.asarray(raw["summary_mask"]).sum(axis=1),
            "DDM_logBF_Mul_Add": ddm_bf,
            "RACE_logBF_Mul_Add": race_bf,
            "delta_DDM_minus_RACE": ddm_bf - race_bf,
        })
        for prefix, pr, names in [("DDM", ddm_p, DDM), ("RACE", race_p, RACE)]:
            for i, label in enumerate(LABELS):
                row[f"{prefix}_p_{label}"] = pr[:, i]
            row[f"{prefix}_best_model"] = np.asarray(names)[pr.argmax(axis=1)]
        frames.append(row)
        print(f"[EVAL] {start+k}/{n}", flush=True)

    frame = pd.concat(frames, ignore_index=True)
    summary = paired_summary(frame, a.bootstrap, a.seed + 900000,
                             generator=a.generator)
    for prefix, names in [("DDM", DDM), ("RACE", RACE)]:
        values = frame[f"{prefix}_best_model"].value_counts()
        summary[prefix]["winner_counts"] = {name:int(values.get(name, 0)) for name in names}
        summary[prefix]["mean_model_probs"] = {
            label:float(frame[f"{prefix}_p_{label}"].mean()) for label in LABELS
        }
    summary["generator_parameter_priors"] = {
        name:[float(z) for z in bounds]
        for name, bounds in zip(model_infos[a.generator]["free_pnames"],
                                model_infos[a.generator]["free_pranges"])
    }
    summary["attention_parameter"] = attention_name
    summary["simulation_counters"] = counters
    summary["input_sha256"] = race_ex.sha256
    summary["checkpoint_DDM"] = str(ddm_folder / "best.keras")
    summary["checkpoint_RACE"] = str(race_folder / "best.keras")
    summary["model_priors"] = [0.25]*4
    frame.to_csv(out / "per_simulation.csv", index=False)
    dump(out / "summary.json", summary)
    quartile_col = f"{attention_name}_quartile"
    frame[quartile_col] = pd.qcut(frame[f"true_{attention_name}"], 4,
                                   duplicates="drop")
    bins = frame.groupby(quartile_col, observed=True).agg(
        n=("sim_id", "size"), mean_attention_param=(f"true_{attention_name}", "mean"),
        ddm_logBF=("DDM_logBF_Mul_Add", "mean"),
        race_logBF=("RACE_logBF_Mul_Add", "mean"),
        paired_delta=("delta_DDM_minus_RACE", "mean"),
    ).reset_index()
    bins[quartile_col] = bins[quartile_col].astype(str)
    bins.to_csv(out / f"{attention_name}_quartiles.csv", index=False)
    plots(frame, out, generator=a.generator)
    print("[RESULT] Mean DDM - RACE log BF =", summary["mean_delta"], flush=True)
    print("[RESULT] Paired t p =", summary["paired_t_p_two_sided"], flush=True)
    print("[RESULT] Bootstrap CI =", summary["mean_delta_bootstrap_ci95"], flush=True)
    print("[OUTPUT]", out, flush=True)


if __name__ == "__main__":
    main()
