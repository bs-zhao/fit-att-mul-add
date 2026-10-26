"""Plot seven SBI classifier comparisons from food_equal and krajbich2010.

Each bar is the participant-averaged softmax probability obtained by
normalizing *within the specified classifier's full candidate model set*.
This is not a Top-1 count, nor a cross-participant joint posterior.

Figure 1 uses simulated aRACE_g data with the food_equal NO-REGEN
two-ACC and two-DDM classifiers, matching s6_decode_maxT_test.py.
Figures 2-7 use real participants and the selected comparison pipeline.

Files are read from the s6_decode_*.py CSV outputs, never retrained here.
If an input is missing or unusable, its subplot remains blank.

Run from any working directory:
    python run/plot/plot_sbi_model_probabilities_7fig.py
    python run/plot/plot_sbi_model_probabilities_7fig.py --pipeline compare_regen
    python run/plot/plot_sbi_model_probabilities_7fig.py --output-dir /path/to/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Work on headless OSC/Slurm nodes.
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
    # The simulated-data figure always uses the explicitly requested
    # compare_noregen/s6_decode_maxT_test.py outputs.
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


def mean_softmax(csv_file: Path, models: tuple[str, ...]) -> tuple[np.ndarray, int] | None:
    if not csv_file.is_file():
        print(f"[MISSING; blank subplot] {csv_file}")
        return None

    try:
        frame = pd.read_csv(csv_file)
        missing = [model for model in models if model not in frame.columns]
        if missing:
            print(f"[INVALID; blank subplot] {csv_file}: missing columns {missing}")
            return None

        logits = frame.loc[:, list(models)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(logits).all(axis=1)
        if not good.all():
            print(f"[WARNING] {csv_file}: dropping {int((~good).sum())} rows with invalid logits")
        logits = logits[good]
        if logits.shape[0] == 0:
            print(f"[EMPTY; blank subplot] {csv_file}")
            return None

        # Stable row-wise softmax; do not average logits first.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        probabilities = exps / exps.sum(axis=1, keepdims=True)
        averages = probabilities.mean(axis=0)
        print(
            f"[OK] {csv_file} | n={logits.shape[0]} | "
            + ", ".join(f"{m}={p:.4f}" for m, p in zip(models, averages))
        )
        return averages, int(logits.shape[0])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[INVALID; blank subplot] {csv_file}: {exc}")
        return None


def draw_panel(
    ax: plt.Axes,
    *,
    dataset: str,
    models: tuple[str, ...],
    pipeline: str,
    family: str | None = None,
    true_model: str | None = None,
) -> None:
    ax.set_title(family or "All eight models", fontsize=13, weight="bold", pad=11)
    csv_file = csv_path(dataset, models, pipeline, true_model=true_model)
    result = mean_softmax(csv_file, models)
    if result is None:
        # Deliberately leave this whole panel empty, without invented bars.
        ax.set_axis_off()
        return

    means, n_subjects = result
    x = np.arange(len(models))
    bars = ax.bar(
        x,
        means,
        color=[COLORS[m.rsplit("_", 1)[-1]] for m in models],
        width=0.66,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_name(m, include_family=family is None) for m in models],
        fontsize=10 if len(models) < 8 else 9,
        rotation=0 if len(models) < 8 else 36,
        ha="center" if len(models) < 8 else "right",
    )
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_ylabel("Mean softmax probability", fontsize=10)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for bar, prob in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{100 * prob:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.text(
        0.99, 0.99, f"n = {n_subjects}",
        ha="right", va="top", transform=ax.transAxes,
        fontsize=9, color="#5C6673",
    )
    if len(models) == 8:
        ax.axvline(3.5, color="#AAB1BC", ls="--", lw=1)
        ax.text(1.5, 1.035, "DDM", ha="center", fontsize=10, weight="bold")
        ax.text(5.5, 1.035, "ACC", ha="center", fontsize=10, weight="bold")


def draw_figure(
    number: int,
    dataset: str,
    model_groups: tuple[tuple[str, ...], ...],
    *,
    pipeline: str,
    output_dir: Path,
    true_model: str | None = None,
) -> None:
    is_eight = len(model_groups) == 1
    fig, axes = plt.subplots(
        1, 1 if is_eight else 2,
        figsize=(11.6, 5.5) if is_eight else (10.2, 4.6),
        squeeze=False,
        facecolor="white",
    )
    axes_list = list(axes.flat)

    if true_model is not None:
        desc = "Simulated data, true model = ACC-Add (aRACE_g)"
    elif number in (2, 3):
        desc = "Real participants; two-model classifiers"
    elif number in (4, 5):
        desc = "Real participants; four-model classifiers"
    else:
        desc = "Real participants; eight-model classifier"

    fig.suptitle(
        f"Figure {number}. {DATASET_NAMES[dataset]} — {desc}",
        fontsize=14, weight="bold", y=0.98,
    )
    for ax, group in zip(axes_list, model_groups):
        family = None if is_eight else ("ACC" if group[0].startswith("aRACE_") else "DDM")
        draw_panel(
            ax,
            dataset=dataset,
            models=group,
            pipeline=pipeline,
            family=family,
            true_model=true_model,
        )

    # Keep an equal probability scale between ACC and DDM panels.
    # Missing CSVs still produce an entirely blank panel.
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.4)
    stem = f"figure_{number}_{dataset}"
    if true_model is not None:
        stem += f"_simulated_{true_model}"
    elif is_eight:
        stem += "_8models"
    else:
        stem += f"_{len(model_groups[0])}models"
    target = output_dir / f"{stem}.png"
    fig.savefig(target, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {target}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        choices=("compare_noregen", "compare_regen"),
        default="compare_noregen",
        help="Source for real-data figures 2-7 (figure 1 always uses compare_noregen).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for all seven figures (PNG and PDF).",
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "outputs" / "plots" / f"sbi_7fig_{args.pipeline}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    specifications = (
        (1, "food_equal", (ACC_2, DDM_2), "aRACE_g"),
        (2, "food_equal", (ACC_2, DDM_2), None),
        (3, "krajbich2010", (ACC_2, DDM_2), None),
        (4, "food_equal", (ACC_4, DDM_4), None),
        (5, "krajbich2010", (ACC_4, DDM_4), None),
        (6, "food_equal", (ALL_8,), None),
        (7, "krajbich2010", (ALL_8,), None),
    )
    for number, dataset, groups, true_model in specifications:
        draw_figure(
            number, dataset, groups,
            pipeline=args.pipeline,
            output_dir=output_dir,
            true_model=true_model,
        )
    print(f"\nComplete: seven PNG figures in {output_dir}")


if __name__ == "__main__":
    main()
