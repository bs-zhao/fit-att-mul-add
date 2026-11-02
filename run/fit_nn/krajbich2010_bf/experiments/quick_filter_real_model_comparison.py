"""Quick real-data filtering sensitivity check using EXISTING S4 classifiers.

No retraining or modification of the S4 simulation prior. For each participant,
drop trials with RT < threshold or fixation-segment count < threshold, then
drop participants with too few remaining trials. Compare baseline predictions
on the SAME retained participants with predictions on their filtered trials.

CAUTION: This is a sensitivity check, not a fully recalibrated Bayes factor:
the S4 networks were trained without the new trial selection mechanism.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from scipy import stats
from _shared.common import (parser, positive, nonnegative, selected_models,
                            load, dump)
from _shared.data import SCALE

RT_IDX = 4
NF_IDX = 9
KEYS = ("trials", "summary_mask", "n_trials")


def filter_real(raw, ids, min_rt_ms=300, min_fixations=2, min_trials=50):
    """Compact remaining trial sets; recompute masks and n_trials correctly.

    All input data come from Experiment.real(), so existing technical filters
    were applied before this additional thresholding.
    """
    audit, retained_indices, rows = [], [], []
    n_max = int(raw["trials"].shape[1])
    for i, sid in enumerate(ids):
        valid = np.asarray(raw["summary_mask"][i], dtype=bool)
        trials = np.asarray(raw["trials"][i])[valid]
        rt_ms = trials[:, RT_IDX].astype(np.float64) * float(SCALE[RT_IDX]) * 1000
        n_fixations = trials[:, NF_IDX].astype(np.float64) * float(SCALE[NF_IDX])
        short_rt = rt_ms + 1e-3 < min_rt_ms
        few_fixations = n_fixations + 1e-4 < min_fixations
        keep = ~(short_rt | few_fixations)
        remaining = trials[keep]
        included = len(remaining) >= min_trials
        audit.append({
            "subj": sid, "n_original": len(trials),
            "n_excluded_short_rt": int(short_rt.sum()),
            "n_excluded_few_fixations": int(few_fixations.sum()),
            "n_excluded_either": int((~keep).sum()),
            "n_filtered": len(remaining), "included": bool(included),
        })
        if not included:
            continue
        retained_indices.append(i)
        compact = np.zeros((n_max, raw["trials"].shape[2]), dtype=np.float32)
        compact[:len(remaining)] = remaining
        mask = np.arange(n_max) < len(remaining)
        rows.append((compact, mask, np.array([np.log1p(len(remaining))], dtype=np.float32)))
    if not rows:
        raise ValueError("No participants remain after filtering; try lower thresholds")
    filtered = {
        "trials": np.stack([r[0] for r in rows]),
        "summary_mask": np.stack([r[1] for r in rows]),
        "n_trials": np.stack([r[2] for r in rows]),
    }
    baseline_matched = {k: np.asarray(raw[k])[retained_indices] for k in KEYS}
    kept_ids = [ids[i] for i in retained_indices]
    return kept_ids, baseline_matched, filtered, pd.DataFrame(audit)


def infer(est, raw, names):
    """Reproduce S6 exactly under its uniform four-model training prior."""
    k = len(names)
    res = est.estimate(conditions=raw, model_prior=np.full(k, 1/k))
    logbf = np.asarray(res["log_bayes_factors"], dtype=float)
    n = len(raw["trials"])
    if logbf.shape != (n, k) or not np.isfinite(logbf).all():
        raise ValueError(f"Unexpected/nonfinite log Bayes factors: {logbf.shape}")
    logits = logbf - logbf.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    probs = weights / weights.sum(axis=1, keepdims=True)
    if "model_probs" in res:
        predicted = np.asarray(res["model_probs"], dtype=float)
        if predicted.shape != probs.shape or not np.allclose(
                probs, predicted, atol=1e-3):
            raise ValueError("Inconsistent model probabilities and log BFs")
    return probs, logbf


def model_table(ids, names, probs):
    out = pd.DataFrame(probs, columns=names)
    out.insert(0, "subj", ids)
    out["best_model"] = np.asarray(names)[probs.argmax(axis=1)]
    out["entropy"] = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), axis=1)
    ddm = np.array([name.startswith("aDDM") for name in names])
    out["p_DDM"] = probs[:, ddm].sum(axis=1)
    out["p_ACC"] = probs[:, ~ddm].sum(axis=1)
    return out


def main():
    p = parser("Quick filtered real-data model comparison with frozen S4 weights",
               classifier=True)
    p.add_argument("--min-rt-ms", type=nonnegative, default=300)
    p.add_argument("--min-fixations", type=positive, default=2)
    p.add_argument("--min-trials", type=positive, default=50)
    args = p.parse_args()
    names = selected_models(args)
    est, ex, folder = load(args, classifier=True)
    ids, raw, original_audit = ex.real()
    kept, before, after, audit = filter_real(
        raw, ids, args.min_rt_ms, args.min_fixations, args.min_trials
    )
    # The SAME frozen classifier evaluates both the original and filtered
    # inputs, on the SAME selected participants.
    baseline_probs, baseline_logbf = infer(est, before, names)
    filtered_probs, filtered_logbf = infer(est, after, names)

    dest = folder / "real_data_filtered" / (
        f"rt{args.min_rt_ms}ms_nf{args.min_fixations}_nt{args.min_trials}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    audit.to_csv(dest / "filter_audit.csv", index=False)
    pd.DataFrame(original_audit).to_csv(dest / "original_trial_audit.csv", index=False)
    model_table(kept, names, baseline_probs).to_csv(
        dest / "model_probabilities_baseline_matched.csv", index=False
    )
    model_table(kept, names, filtered_probs).to_csv(
        dest / "model_probabilities_filtered.csv", index=False
    )

    pairwise = []
    for index, sid in enumerate(kept):
        for m, numerator in enumerate(names):
            for j, denominator in enumerate(names):
                if m != j:
                    pairwise.append({
                        "subj": sid, "numerator": numerator, "denominator": denominator,
                        "log_bayes_factor": float(filtered_logbf[index, m] -
                                                  filtered_logbf[index, j]),
                    })
    pd.DataFrame(pairwise).to_csv(
        dest / "pairwise_log_bayes_factors_filtered.csv", index=False
    )

    compare = pd.DataFrame({"subj": kept})
    counts = audit.set_index("subj").loc[kept]
    compare["original_trials"] = counts["n_original"].to_numpy()
    compare["filtered_trials"] = counts["n_filtered"].to_numpy()
    compare["baseline_best"] = np.asarray(names)[baseline_probs.argmax(axis=1)]
    compare["filtered_best"] = np.asarray(names)[filtered_probs.argmax(axis=1)]
    if len(names) == 4 and names[2].endswith("_t") and names[3].endswith("_g"):
        mul_i, add_i = 2, 3
    else:
        mul_names = [i for i, m in enumerate(names) if m.endswith("_t")]
        add_names = [i for i, m in enumerate(names) if m.endswith("_g")]
        if len(mul_names) != 1 or len(add_names) != 1:
            raise ValueError("This diagnostic needs one _t (Mul) and one _g (Add) model")
        mul_i, add_i = mul_names[0], add_names[0]

    lbf_before = baseline_logbf[:, mul_i] - baseline_logbf[:, add_i]
    lbf_after = filtered_logbf[:, mul_i] - filtered_logbf[:, add_i]
    compare["baseline_logBF_Mul_Add"] = lbf_before
    compare["filtered_logBF_Mul_Add"] = lbf_after
    compare["delta_logBF_filtered_minus_baseline"] = lbf_after - lbf_before
    compare["baseline_p_Add"] = baseline_probs[:, add_i]
    compare["filtered_p_Add"] = filtered_probs[:, add_i]
    compare.to_csv(dest / "per_subject_comparison.csv", index=False)

    def describe(x, probabilities, best):
        return {
            "mean_logBF_Mul_Add": float(np.mean(x)),
            "median_logBF_Mul_Add": float(np.median(x)),
            "Mul_above_Add": int((x > 0).sum()),
            "Add_above_Mul": int((x < 0).sum()),
            "BF_gt_10_for_Mul": int((x > np.log(10)).sum()),
            "BF_gt_10_for_Add": int((x < -np.log(10)).sum()),
            "mean_Add_PMP": float(np.mean(probabilities[:, add_i])),
            "winning_models": {name: int((best == name).sum()) for name in names},
        }

    summary = {
        "analysis_type": "frozen-classifier quick filtering sensitivity check",
        "warning": "S4 and S5 were NOT retrained or recalibrated for filtered trials",
        "model_names": names,
        "selection": {"min_rt_ms": args.min_rt_ms,
                      "min_fixations": args.min_fixations,
                      "min_trials": args.min_trials},
        "participants_before": len(ids),
        "participants_after": len(kept),
        "trials_before_all": int(audit["n_original"].sum()),
        "trials_after_all": int(audit["n_filtered"].sum()),
        "trials_after_included": int(counts["n_filtered"].sum()),
        "baseline_matched": describe(
            lbf_before, baseline_probs, compare["baseline_best"].to_numpy()),
        "filtered": describe(
            lbf_after, filtered_probs, compare["filtered_best"].to_numpy()),
        "mean_delta_logBF_Mul_Add": float(np.mean(lbf_after - lbf_before)),
        "n_subjects_logBF_increased": int(((lbf_after-lbf_before) > 0).sum()),
        "n_subjects_logBF_decreased": int(((lbf_after-lbf_before) < 0).sum()),
        "checkpoint": str(folder / "best.keras"),
        "input_sha256": ex.sha256,
    }
    if len(kept) >= 2:
        test = stats.ttest_rel(lbf_after, lbf_before)
        summary["paired_t_test"] = {"t": float(test.statistic),
                                     "p_two_sided": float(test.pvalue)}
    dump(dest / "summary.json", summary)
    print("[FILTER]", args.min_rt_ms, "ms", args.min_fixations,
          "fixations", args.min_trials, "trials minimum", flush=True)
    print("[PARTICIPANTS]", len(ids), "->", len(kept), flush=True)
    print("[TRIALS]", summary["trials_before_all"], "->",
          summary["trials_after_included"], flush=True)
    print("[MUL/ADD mean log BF] matched baseline:",
          summary["baseline_matched"]["mean_logBF_Mul_Add"],
          "filtered:", summary["filtered"]["mean_logBF_Mul_Add"], flush=True)
    print("[WINNERS] original:", summary["baseline_matched"]["winning_models"],
          "filtered:", summary["filtered"]["winning_models"], flush=True)
    print("[OUTPUT]", dest, flush=True)


if __name__ == "__main__":
    main()
