"""Create three editor-letter figures from existing model-classifier logits.

Kept from compose_mean_prob_paper_figs.py:
    Original Fig. 1: Food Equal, simulated ACC-Add data, four ACC vs four DDM.
    Original Fig. 2: Food Equal, observed data, four ACC / four DDM / eight.
    Original Fig. 4: Krajbich et al. (2010), observed data, eight-model panel ONLY.

Every panel reads its own classifier CSV. For each participant, softmax is
calculated over that classifier's candidates; the plotted quantity is the mean
of those softmax scores across participants. Values are NOT BIC weights or
validated Bayesian posterior model probabilities.

Usage (from any working directory):
    python run/plot/compose_mean_prob_editor_letter.py --pipeline compare_regen
    python run/plot/compose_mean_prob_editor_letter.py --pipeline compare_noregen

Outputs: outputs/plots/editor_letter_mean_prob/<pipeline>/
    figure_1_simulated_ACC_Add_food_equal.png
    figure_2_observed_food_equal.png
    figure_3_observed_krajbich_eight_models.png
    plotted_values.csv

No training, decoding, CSV modification, or simulated data generation occurs.
All required source CSVs must be present; missing/invalid inputs fail loudly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ACC_4 = ("aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g")
DDM_4 = ("aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g")
ALL_8 = ACC_4 + DDM_4

LABELS = {"1": "NoAtt", "2": "AttOnly", "t": "Mul", "g": "Add"}
SHORT = {"1": "NoAtt", "2": "AttOnly", "t": "Mul", "g": "Add"}
COLORS = {"1": "#87919D", "2": "#C89C55", "t": "#386FA4", "g": "#C9655D"}
INK = "#233044"
MUTED = "#5C6877"
GRID = "#E8EDF2"


def setup_style():
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "axes.edgecolor": "#CBD2DA", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": MUTED,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def csv_location(root: Path, dataset: str, pipeline: str, models: tuple,
                 simulated: bool) -> Path:
    branch = ("test/trial/s6_decode_trial/aRACE_g" if simulated
              else "real_data/trial/s6_decode_trial")
    # The existing eight-model decoder saves filenames in DDM-then-ACC order.
    # Keep figure labels in ACC-then-DDM order; read_scores selects columns by name.
    file_models = DDM_4 + ACC_4 if models == ALL_8 else models
    return root / "outputs" / dataset / pipeline / branch / ("_".join(file_models) + ".csv")


def read_scores(path: Path, models: tuple) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required classifier output not found: {path}\n"
                                "Run the corresponding s6 decoding script first.")
    frame = pd.read_csv(path)
    missing = [name for name in models if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    logits = frame.loc[:, models].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError(f"No participants in {path}")
    if not np.isfinite(logits).all():
        raise ValueError(f"Nonfinite logits detected in {path}; refusing to hide rows.")
    # Numerically stable participant-wise softmax, THEN average participants.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    means = probabilities.mean(axis=0)
    if not np.isclose(means.sum(), 1.0, atol=1e-7):
        raise AssertionError(f"Softmax probabilities do not sum to one: {path}")
    print(f"[READ] n={len(logits):3d} | {path}")
    print("       " + ", ".join(f"{m}={v:.4f}" for m, v in zip(models, means)))
    return means, len(logits)


def model_labels(models: tuple, family_labels: bool = False) -> list[str]:
    if family_labels:
        return [("ACC" if m.startswith("aRACE") else "DDM") + "-" + SHORT[m.rsplit("_", 1)[1]]
                for m in models]
    return [LABELS[m.rsplit("_", 1)[1]] for m in models]


def bars(ax, models: tuple, means: np.ndarray, n: int, title: str,
         *, compact: bool = False):
    x = np.arange(len(models))
    colors = [COLORS[m.rsplit("_", 1)[1]] for m in models]
    rects = ax.bar(x, means, color=colors, width=0.66 if not compact else 0.62,
                   edgecolor="white", linewidth=0.8, zorder=3)
    if title:
        ax.set_title(title, loc="left", pad=15, fontweight="bold", color=INK)
    ax.set_ylabel("Mean softmax probability", labelpad=9)
    ax.set_ylim(0, 1.045)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#D3D9E0")
    labels = model_labels(models, family_labels=compact)
    ax.set_xticks(x, labels, rotation=0 if compact else 15,
                  ha="center" if compact else "right", fontsize=9)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0)
    ax.text(0.99, 0.985, f"n = {n}", transform=ax.transAxes, ha="right",
            va="top", color=MUTED, fontsize=9)
    for rect, val in zip(rects, means):
        ax.annotate(f"{val:.1%}", (rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9, color=INK)
    if compact:
        ax.axvline(3.5, color="#BFC7CF", linestyle="--", linewidth=1.0,
                   zorder=2)
    ax.margins(x=0.035)


def heading(fig, title: str):
    fig.text(0.055, 0.965, title, fontsize=13, fontweight="bold",
             color=INK, ha="left", va="top")


def section_heading(fig, title: str, y: float):
    """Equal typographic level for the two comparisons in Figure 2."""
    fig.text(0.075, y, title, fontsize=12, fontweight="bold",
             color=INK, ha="left", va="top")


def write(fig, outdir: Path, name: str, dpi: int):
    dest = outdir / f"{name}.png"
    fig.savefig(dest, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print(f"[SAVED] {dest}")
    plt.close(fig)


def record(metrics: list[dict], figure: str, panel: str, dataset: str,
           kind: str, path: Path, models: tuple, values: np.ndarray, n: int):
    for model, value in zip(models, values):
        metrics.append({"figure": figure, "panel": panel, "dataset": dataset,
                        "data_type": kind, "true_model": "aRACE_g" if kind == "simulated" else "",
                        "n_participants": n, "candidate_model": model,
                        "mean_softmax_probability": f"{value:.8f}", "source_csv": str(path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("compare_regen", "compare_noregen"),
                        default="compare_regen")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    outdir = (args.output_dir.expanduser().resolve() if args.output_dir else
              root / "outputs/plots/editor_letter_mean_prob" / args.pipeline)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_style()
    metrics: list[dict] = []

    # Figure 1: original Figure 1 (all content) -- Food Equal simulated ACC-Add.
    fig = plt.figure(figsize=(10.6, 4.6))
    heading(fig, "ACC-Add simulations")
    ax1 = fig.add_axes([0.075, 0.17, 0.400, 0.63])
    ax2 = fig.add_axes([0.575, 0.17, 0.400, 0.63])
    for ax, models, family in [(ax1, ACC_4, "ACC"),
                                (ax2, DDM_4, "DDM")]:
        path = csv_location(root, "food_equal", args.pipeline, models, simulated=True)
        values, n = read_scores(path, models)
        bars(ax, models, values, n, family)
        record(metrics, "1", family, "food_equal", "simulated", path, models, values, n)
    write(fig, outdir, "figure_1_simulated_ACC_Add_food_equal", args.dpi)

    # Figure 2: original Figure 2 (all content) -- current-study observed data.
    fig = plt.figure(figsize=(10.6, 7.6))
    section_heading(fig, "Within-model-family comparison (4-class classifiers)", 0.945)
    section_heading(fig, "Cross-family comparison (8-class classifier)", 0.475)
    ax1 = fig.add_axes([0.075, 0.54, 0.400, 0.29])
    ax2 = fig.add_axes([0.575, 0.54, 0.400, 0.29])
    ax3 = fig.add_axes([0.085, 0.12, 0.890, 0.29])
    for ax, models, family, compact in [
        (ax1, ACC_4, "ACC", False),
        (ax2, DDM_4, "DDM", False),
        (ax3, ALL_8, "", True),
    ]:
        path = csv_location(root, "food_equal", args.pipeline, models, simulated=False)
        values, n = read_scores(path, models)
        bars(ax, models, values, n, family, compact=compact)
        record(metrics, "2", family, "food_equal", "observed", path, models, values, n)
    write(fig, outdir, "figure_2_observed_food_equal", args.dpi)

    # Figure 3: ONLY bottom eight-model panel of original Figure 4.
    fig = plt.figure(figsize=(10.6, 4.5))
    heading(fig, "Krajbich et al. (2010)")
    ax = fig.add_axes([0.085, 0.18, 0.890, 0.63])
    path = csv_location(root, "krajbich2010", args.pipeline, ALL_8, simulated=False)
    values, n = read_scores(path, ALL_8)
    bars(ax, ALL_8, values, n, "ACC + DDM", compact=True)
    record(metrics, "3", "ACC + DDM",
           "krajbich2010", "observed", path, ALL_8, values, n)
    write(fig, outdir, "figure_3_observed_krajbich_eight_models", args.dpi)

    fields = ["figure", "panel", "dataset", "data_type", "true_model",
              "n_participants", "candidate_model", "mean_softmax_probability", "source_csv"]
    with (outdir / "plotted_values.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"[SAVED] {outdir / 'plotted_values.csv'}")
    print("Completed: 3 PNG figures and plotted_values.csv.")


if __name__ == "__main__":
    main()
