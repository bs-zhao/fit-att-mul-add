"""Nine SBI model-comparison figures with selectable summary metrics.

Figure 1: food_equal, simulated ACC-Add (aRACE_g); two ACC vs two DDM
          compare_noregen classifiers.
Figures 2-3: real food_equal / krajbich2010, two ACC vs two DDM.
Figures 4-5: real food_equal / krajbich2010, four ACC vs four DDM.
Figures 6-7: real food_equal / krajbich2010, joint eight-model classifier.
Figure 8: food_equal, simulated ACC-Add (aRACE_g); four ACC vs four DDM
          compare_noregen classifiers (the four-model counterpart of Figure 1).
Figure 9: food_equal, real participants; four ACC vs four DDM
          (the four-model counterpart of Figure 2; duplicates Figure 4).

The source CSVs contain *logits*. Softmax is applied separately to each
subject's full classifier candidate set, before any across-subject summary.
A full-model winner differs from a within-architecture Mul-vs-Add winner.
The latter is intentionally calculated from just the two relevant logits,
even in the four- and eight-model comparisons.

Figures are saved as PNG only. A missing / unusable CSV leaves its panel
blank. The program never substitutes other data or retrains a classifier.

Examples (run from any working directory):
    python run/plot/plot_sbi_model_probabilities_7fig.py
    python run/plot/plot_sbi_model_probabilities_7fig.py --metric winner_count
    python run/plot/plot_sbi_model_probabilities_7fig.py --metric pairwise_prob
    python run/plot/plot_sbi_model_probabilities_7fig.py --metric logit_sum
    python run/plot/plot_sbi_model_probabilities_7fig.py --metric all
    # Without --metric: generate all metrics, one folder per metric.
    python run/plot/plot_sbi_model_probabilities_7fig.py --pipeline compare_regen --metric mean_prob

Metric definitions (s: subject, m: model, z_sm: classifier logit,
p_sm: softmax across ALL candidate models of this classifier):
  mean_prob          average_s p_sm
  median_prob        median_s p_sm
  prob_sum           sum_s p_sm (expected number of model assignments)
  winner_count       sum_s 1[m = argmax_k z_sk]
  winner_pct         100 * winner_count / number of subjects
  pairwise_prob      average_s softmax(z_s,Mul, z_s,Add), within each family
  pairwise_win_count sum_s 1[z_s,m > z_s,other], Mul vs Add only
  pairwise_win_pct   100 * pairwise_win_count / number of subjects
  mean_logit         average_s z_sm (diagnostic, NOT a posterior probability)
  logit_sum          sum_s z_sm (diagnostic, NOT a posterior probability)

Pairwise metrics show only Mul and Add; NoAtt and AttOnly are omitted
from those bar charts, but their logits remain part of the original
classifier training and full softmax definition. Exact ties count for
neither side in pairwise_win_count or pairwise_win_pct.

These are descriptive per-subject summaries, not a group-level
joint Bayesian posterior; classifier probabilities require calibration
before being treated as reliable model posteriors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Supports OSC/Slurm nodes without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

ACC_2 = ("aRACE_t", "aRACE_g")
DDM_2 = ("aDDM_t", "aDDM_g")
ACC_4 = ("aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g")
DDM_4 = ("aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g")
ALL_8 = DDM_4 + ACC_4

MECHANISM_NAMES = {"1": "NoAtt", "2": "AttOnly", "t": "Mul", "g": "Add"}
COLORS = {
    "1": "#8994A4",
    "2": "#E6A33E",
    "t": "#4778BD",
    "g": "#D46D67",
}
DATASET_NAMES = {"food_equal": "Food equal", "krajbich2010": "Krajbich (2010)"}

# One CLI parameter controls the metric for *all seven* figures.
METRIC_DESCRIPTIONS = {
    "mean_prob": "Mean softmax probability (default)",
    "median_prob": "Median softmax probability",
    "prob_sum": "Sum of softmax probabilities across subjects",
    "winner_count": "Number of participants with this model as overall Top-1",
    "winner_pct": "Percentage of participants with this model as overall Top-1",
    "pairwise_prob": "Mean conditional Mul-vs-Add softmax, within each architecture",
    "pairwise_win_count": "Number of participants preferring Mul or Add in the pair",
    "pairwise_win_pct": "Percentage of participants preferring Mul or Add in the pair",
    "mean_logit": "Mean raw classifier logit (diagnostic only)",
    "logit_sum": "Sum of raw classifier logits (diagnostic only)",
}
PAIRWISE_METRICS = {"pairwise_prob", "pairwise_win_count", "pairwise_win_pct"}
COUNT_METRICS = {"winner_count", "pairwise_win_count"}
PCT_METRICS = {"winner_pct", "pairwise_win_pct"}
PROB_METRICS = {"mean_prob", "median_prob", "pairwise_prob"}
SIGNED_METRICS = {"mean_logit", "logit_sum"}

Y_LABELS = {
    "mean_prob": "Mean softmax probability",
    "median_prob": "Median softmax probability",
    "prob_sum": "Sum of softmax probabilities",
    "winner_count": "Top-1 winner count",
    "winner_pct": "Top-1 winner proportion (%)",
    "pairwise_prob": "Mean conditional P(model | Mul or Add)",
    "pairwise_win_count": "Pairwise winner count",
    "pairwise_win_pct": "Pairwise winner proportion (%)",
    "mean_logit": "Mean raw logit (not probability)",
    "logit_sum": "Sum of raw logits (not probability)",
}


def display_name(model: str, *, include_family: bool = False) -> str:
    family = "ACC" if model.startswith("aRACE_") else "DDM"
    mechanism = MECHANISM_NAMES[model.rsplit("_", 1)[-1]]
    return f"{family}-{mechanism}" if include_family else mechanism


def csv_path(
    dataset: str,
    models: tuple[str, ...],
    pipeline: str,
    *,
    true_model: str | None = None,
) -> Path:
    # Figure 1 always uses the specific compare_noregen test-data output.
    source = "compare_noregen" if true_model is not None else pipeline
    sample = "test" if true_model is not None else "real_data"
    directory = (
        REPO_ROOT
        / "outputs"
        / dataset
        / source
        / sample
        / "trial"
        / "s6_decode_trial"
    )
    if true_model is not None:
        directory = directory / true_model
    return directory / ("_".join(models) + ".csv")


def load_logits(csv_file: Path, models: tuple[str, ...]) -> np.ndarray | None:
    if not csv_file.is_file():
        print(f"[MISSING; blank subplot] {csv_file}")
        return None

    try:
        frame = pd.read_csv(csv_file)
        missing = [m for m in models if m not in frame.columns]
        if missing:
            print(f"[INVALID; blank subplot] {csv_file}: missing columns {missing}")
            return None

        logits = frame.loc[:, list(models)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        good = np.isfinite(logits).all(axis=1)
        if not good.all():
            print(f"[WARNING] {csv_file}: dropping {(~good).sum()} invalid rows")
        logits = logits[good]
        if len(logits) == 0:
            print(f"[EMPTY; blank subplot] {csv_file}")
            return None
        return logits
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[INVALID; blank subplot] {csv_file}: {exc}")
        return None


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Stable within-subject softmax over the supplied logits' columns."""
    centered = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(centered)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def metric_values(
    logits: np.ndarray, models: tuple[str, ...], metric: str
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return bar model IDs and corresponding values.

    The full candidate set is used for mean_prob and winner_count.
    For pairwise_* metrics, only Mul and Add compete, separately
    inside ACC and DDM, even when the full classifier has 8 models.
    """
    n, num_models = logits.shape
    if metric in PAIRWISE_METRICS:
        by_family = [
            family for family in ("aRACE", "aDDM")
            if f"{family}_t" in models and f"{family}_g" in models
        ]
        chosen_models: list[str] = []
        chosen_values: list[float] = []
        for family in by_family:
            mul, add = f"{family}_t", f"{family}_g"
            pair = logits[:, [models.index(mul), models.index(add)]]
            if metric == "pairwise_prob":
                values = softmax_rows(pair).mean(axis=0)
            else:
                counts = np.array([
                    np.count_nonzero(pair[:, 0] > pair[:, 1]),
                    np.count_nonzero(pair[:, 1] > pair[:, 0]),
                ], dtype=float)
                values = counts if metric == "pairwise_win_count" else counts * (100.0 / n)
            chosen_models.extend((mul, add))
            chosen_values.extend(values.tolist())
        return tuple(chosen_models), np.asarray(chosen_values, dtype=float)

    if metric in ("mean_prob", "median_prob", "prob_sum"):
        probabilities = softmax_rows(logits)
        if metric == "mean_prob":
            values = probabilities.mean(axis=0)
        elif metric == "median_prob":
            values = np.median(probabilities, axis=0)
        else:
            values = probabilities.sum(axis=0)
    elif metric in ("winner_count", "winner_pct"):
        counts = np.bincount(np.argmax(logits, axis=1), minlength=num_models).astype(float)
        values = counts if metric == "winner_count" else counts * (100.0 / n)
    elif metric == "mean_logit":
        values = logits.mean(axis=0)
    elif metric == "logit_sum":
        values = logits.sum(axis=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return models, np.asarray(values, dtype=float)


def bar_label(value: float, metric: str) -> str:
    if metric in COUNT_METRICS:
        return f"{int(round(value))}"
    if metric in PCT_METRICS:
        return f"{value:.1f}%"
    if metric in PROB_METRICS:
        return f"{100 * value:.1f}%"
    return f"{value:.2f}"


def draw_panel(
    ax: plt.Axes,
    *,
    dataset: str,
    models: tuple[str, ...],
    pipeline: str,
    metric: str,
    family: str | None = None,
    true_model: str | None = None,
) -> None:
    ax.set_title(family or "All eight models", fontsize=13, weight="bold", pad=11)
    csv_file = csv_path(dataset, models, pipeline, true_model=true_model)
    logits = load_logits(csv_file, models)
    if logits is None:
        # Required behavior: leave only this subplot blank.
        ax.set_axis_off()
        return

    selected_models, values = metric_values(logits, models, metric)
    if family is None:
        # In the joint comparison, DISPLAY ACC on the left and DDM on the
        # right without changing the CSV filename, source-column order,
        # full-set softmax, or winner calculations.
        order = sorted(
            range(len(selected_models)),
            key=lambda i: (not selected_models[i].startswith("aRACE_"), i),
        )
        selected_models = tuple(selected_models[i] for i in order)
        values = values[order]
    n = len(logits)
    print(
        f"[OK] {csv_file} | metric={metric} | n={n} | "
        + ", ".join(f"{m}={v:.5g}" for m, v in zip(selected_models, values))
    )

    x = np.arange(len(selected_models))
    bars = ax.bar(
        x, values, width=0.66, edgecolor="white", linewidth=0.7,
        color=[COLORS[m.rsplit("_", 1)[-1]] for m in selected_models],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_name(m, include_family=family is None) for m in selected_models],
        fontsize=10 if len(selected_models) < 8 else 9,
        rotation=0 if len(selected_models) < 8 else 36,
        ha="center" if len(selected_models) < 8 else "right",
    )
    ax.set_ylabel(Y_LABELS[metric], fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.22)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if metric in PROB_METRICS:
        upper, lower = 1.08, 0.0
        ticks = np.linspace(0, 1, 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(round(v * 100))}%" for v in ticks])
    elif metric in PCT_METRICS:
        upper, lower = 108.0, 0.0
        ticks = np.linspace(0, 100, 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(round(v))}%" for v in ticks])
    elif metric == "prob_sum" or metric in COUNT_METRICS:
        lower, upper = 0.0, max(1.0, n * 1.08)
    else:
        # Raw logits can be negative. Keep zero visible and do not
        # misleadingly label the y-axis with percentages.
        low_val, high_val = min(0.0, float(values.min())), max(0.0, float(values.max()))
        span = max(1.0, high_val - low_val)
        lower = low_val - (0.14 * span if low_val < 0 else 0.0)
        upper = high_val + 0.18 * span
    ax.set_ylim(lower, upper)

    pad = 0.017 * (upper - lower)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + (pad if value >= 0 else -pad),
            bar_label(float(value), metric),
            ha="center", va="bottom" if value >= 0 else "top", fontsize=9,
        )
    ax.text(
        0.99, 0.99, f"n = {n}",
        ha="right", va="top", transform=ax.transAxes,
        fontsize=9, color="#5C6673",
    )

    # Figure 6/7: separate ACC (left) and DDM (right) within either the full eight bars
    # or the four bars for the pairwise-only metrics.
    if family is None and len(selected_models) in (4, 8):
        split = 1.5 if len(selected_models) == 4 else 3.5
        left = 0.5 if len(selected_models) == 4 else 1.5
        right = 2.5 if len(selected_models) == 4 else 5.5
        ax.axvline(split, color="#AAB1BC", ls="--", lw=1)
        for pos, text_label in ((left, "ACC"), (right, "DDM")):
            ax.text(
                pos, 1.025, text_label, ha="center",
                transform=ax.get_xaxis_transform(),
                fontsize=10, weight="bold", clip_on=False,
            )


def draw_figure(
    number: int,
    dataset: str,
    model_groups: tuple[tuple[str, ...], ...],
    *,
    pipeline: str,
    metric: str,
    output_dir: Path,
    true_model: str | None = None,
) -> None:
    is_eight = len(model_groups) == 1
    fig, axes = plt.subplots(
        1, 1 if is_eight else 2,
        figsize=(11.6, 5.5) if is_eight else (10.2, 4.6),
        squeeze=False, facecolor="white",
    )

    if true_model is not None:
        desc = "Simulated data, true model = ACC-Add (aRACE_g)"
    elif number in (2, 3):
        desc = "Real participants; two-model classifiers"
    elif number in (4, 5, 9):
        desc = "Real participants; four-model classifiers"
    else:
        desc = "Real participants; eight-model classifier"

    fig.suptitle(
        f"Figure {number}. {DATASET_NAMES[dataset]} — {desc}\n"
        f"{METRIC_DESCRIPTIONS[metric]}",
        fontsize=13, weight="bold", y=0.99,
    )
    for ax, models in zip(axes.flat, model_groups):
        family = None if is_eight else ("ACC" if models[0].startswith("aRACE_") else "DDM")
        draw_panel(
            ax, dataset=dataset, models=models, pipeline=pipeline,
            metric=metric, family=family, true_model=true_model,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.4)
    stem = f"figure_{number}_{dataset}"
    if true_model is not None:
        stem += f"_simulated_{true_model}"
    elif is_eight:
        stem += "_8models"
    else:
        stem += f"_{len(model_groups[0])}models"
    target = output_dir / f"{stem}_{metric}.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {target}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metric", choices=tuple(METRIC_DESCRIPTIONS) + ("all",),
        default="all",
        help="Plotted statistic; default 'all' generates nine figures per statistic.",
    )
    parser.add_argument(
        "--pipeline", choices=("compare_noregen", "compare_regen"),
        default="compare_noregen",
        help="Source for real-data figures 2-7 and 9; figures 1 and 8 always use compare_noregen.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Destination directory for PNG figures only.",
    )
    args = parser.parse_args()
    output_base = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None
        else REPO_ROOT / "outputs" / "plots" / f"sbi_9fig_{args.pipeline}"
    )

    specifications = (
        (1, "food_equal", (ACC_2, DDM_2), "aRACE_g"),
        (2, "food_equal", (ACC_2, DDM_2), None),
        (3, "krajbich2010", (ACC_2, DDM_2), None),
        (4, "food_equal", (ACC_4, DDM_4), None),
        (5, "krajbich2010", (ACC_4, DDM_4), None),
        (6, "food_equal", (ALL_8,), None),
        (7, "krajbich2010", (ALL_8,), None),
        (8, "food_equal", (ACC_4, DDM_4), "aRACE_g"),
        (9, "food_equal", (ACC_4, DDM_4), None),
    )
    metrics = tuple(METRIC_DESCRIPTIONS) if args.metric == "all" else (args.metric,)
    for metric in metrics:
        metric_dir = output_base / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== Metric: {metric} -> {metric_dir} =====")
        for number, dataset, groups, true_model in specifications:
            draw_figure(
                number, dataset, groups, pipeline=args.pipeline,
                metric=metric, output_dir=metric_dir, true_model=true_model,
            )
    print(f"\nComplete: {len(metrics) * len(specifications)} PNG figure(s) under {output_base}")


if __name__ == "__main__":
    main()
