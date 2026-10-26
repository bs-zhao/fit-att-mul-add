"""Compose four publication-style mean_prob figures for one chosen pipeline.

All three figures are rendered directly from the selected pipeline's classifier
logit CSVs, with mean_prob calculated via per-participant softmax followed by
participant averaging. No source PNGs are cropped, masked, or rescaled.

Example:
    python run/plot/compose_mean_prob_paper_figs.py --pipeline compare_regen
    python run/plot/compose_mean_prob_paper_figs.py --pipeline compare_noregen

Source real CSVs: outputs/<dataset>/<pipeline>/real_data/trial/s6_decode_trial/
Source simulated CSVs: outputs/<dataset>/<pipeline>/test/trial/s6_decode_trial/aRACE_g/
Default outputs: outputs/plots/mean_prob_paper_figs/<pipeline>/

This script NEVER trains or runs a decoder. Missing or invalid input files
leave only the corresponding subplot blank; the other panels still render.
"""

from pathlib import Path
import argparse
from io import BytesIO

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ACC_4 = ("aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g")
DDM_4 = ("aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g")
MODEL_LABELS = {"1": "NoAtt", "2": "AttOnly", "t": "Mul", "g": "Add"}
MODEL_COLORS = {"1": "#8994A4", "2": "#E6A33E", "t": "#4778BD", "g": "#D46D67"}


def crop_whitespace(img, threshold=245, pad=8):
    arr = np.asarray(img.convert("RGB"))
    mask = (arr < threshold).any(axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    x0, x1 = max(0, xs.min()-pad), min(img.width, xs.max()+pad+1)
    y0, y1 = max(0, ys.min()-pad), min(img.height, ys.max()+pad+1)
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def mask_top_title(img, frac=0.10):
    out = img.convert("RGB").copy()
    ImageDraw.Draw(out).rectangle(
        (0, 0, out.width, int(out.height * frac)), fill="white"
    )
    return out


def make_placeholder(text, size=(1800, 1000)):
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((80, 80), text, fill=(80, 80, 80))
    return img


def load_panel(path, title_mask_frac=0.10):
    if not path.is_file():
        print(f"[MISSING] {path}")
        return make_placeholder(f"Missing: {path.name}")
    with Image.open(path) as img:
        panel = crop_whitespace(img.convert("RGB"))
    return crop_whitespace(mask_top_title(panel, frac=title_mask_frac))



def simulated_csv_dir(repo_root, dataset, pipeline):
    """Location of test logits from the requested dataset and pipeline."""
    return (
        repo_root / "outputs" / dataset / pipeline
        / "test" / "trial" / "s6_decode_trial" / "aRACE_g"
    )


def render_simulated_panel(repo_root, dataset, pipeline):
    """Replot one dataset's aRACE_g test data using its own ACC/DDM classifiers.

    Never mix a dataset or a pipeline across true data and classifier output.
    Each set of 4 models is normalized separately, as in the original Figure 8.
    """
    source = simulated_csv_dir(repo_root, dataset, pipeline)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), facecolor="white")
    for ax, models, family in zip(axes, (ACC_4, DDM_4), ("ACC", "DDM")):
        ax.set_title(family, fontsize=13, weight="bold", pad=11)
        path = source / ("_".join(models) + ".csv")
        if not path.is_file():
            print(f"[MISSING; blank panel] {path}")
            ax.set_axis_off()
            continue
        try:
            frame = pd.read_csv(path)
            missing = [m for m in models if m not in frame.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            logits = frame.loc[:, list(models)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(logits).all(axis=1)
            if not valid.all():
                print(f"[WARNING] {path}: skipped {(~valid).sum()} invalid rows")
            logits = logits[valid]
            if not len(logits):
                raise ValueError("No valid rows")
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            means = (exp_scores / exp_scores.sum(axis=1, keepdims=True)).mean(axis=0)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[INVALID; blank panel] {path}: {exc}")
            ax.set_axis_off()
            continue
        print(f"[OK] {path}: n={len(logits)} | " +
              ", ".join(f"{m}={v:.4f}" for m, v in zip(models, means)))
        xs = np.arange(len(models))
        bars = ax.bar(
            xs, means, width=0.66, linewidth=0.7, edgecolor="white",
            color=[MODEL_COLORS[m.rsplit("_", 1)[-1]] for m in models],
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABELS[m.rsplit("_", 1)[-1]] for m in models])
        ax.set_ylim(0, 1.08)
        ticks = np.linspace(0, 1, 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{round(100*t):d}%" for t in ticks])
        ax.set_ylabel("Mean softmax probability", fontsize=10)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width()/2, float(value)+0.018,
                f"{100*value:.1f}%", ha="center", va="bottom", fontsize=9,
            )
        ax.text(
            0.99, 0.99, f"n = {len(logits)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#5C6673",
        )
    fig.tight_layout(w_pad=2.4)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return crop_whitespace(image.convert("RGB"))


def render_real_figure(repo_root, dataset, pipeline, output_path):
    """Render 4-model comparison above joint 8-model comparison at native quality.

    Direct rendering avoids loss of axis labels during image masking/re-cropping.
    Full-softmax normalization is performed over 4 or 8 classes as appropriate.
    Missing/invalid CSV leaves only its individual subplot blank.
    """
    dataset_label = "Food Equal" if dataset == "food_equal" else "Krajbich et al. (2010)"
    source = (
        repo_root / "outputs" / dataset / pipeline
        / "real_data" / "trial" / "s6_decode_trial"
    )
    fig = plt.figure(figsize=(11.2, 8.4), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.10), hspace=0.19, wspace=0.16)
    ax_acc = fig.add_subplot(grid[0, 0])
    ax_ddm = fig.add_subplot(grid[0, 1])
    ax_all = fig.add_subplot(grid[1, :])
    fig.suptitle(dataset_label, fontsize=15, weight="bold")

    def draw(ax, models, family):
        file = source / ("_".join(models) + ".csv")
        ax.set_title(family or "All eight models", fontsize=12, weight="bold", pad=12)
        if not file.is_file():
            print(f"[MISSING; blank panel] {file}")
            ax.set_axis_off()
            return
        try:
            frame = pd.read_csv(file)
            absent = [name for name in models if name not in frame.columns]
            if absent:
                raise ValueError(f"Missing columns {absent}")
            logits = frame.loc[:, list(models)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            good = np.isfinite(logits).all(axis=1)
            if not good.all():
                print(f"[WARNING] {file}: skipped {(~good).sum()} invalid rows")
            logits = logits[good]
            if logits.shape[0] == 0:
                raise ValueError("No finite participant logits")
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            means = (weights / weights.sum(axis=1, keepdims=True)).mean(axis=0)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[INVALID; blank panel] {file}: {exc}")
            ax.set_axis_off()
            return

        if family is None:
            order = tuple(models.index(name) for name in ACC_4 + DDM_4)
            display_models = ACC_4 + DDM_4
            means = means[list(order)]
        else:
            display_models = models
        print(
            f"[OK] {file}: n={len(logits)} | "
            + ", ".join(f"{name}={v:.5f}" for name, v in zip(display_models, means))
        )
        x = np.arange(len(display_models))
        bars = ax.bar(
            x, means, width=0.65, edgecolor="white", linewidth=0.7,
            color=[MODEL_COLORS[name.rsplit("_", 1)[-1]]
                   for name in display_models],
        )
        labels = [
            MODEL_LABELS[name.rsplit("_", 1)[-1]]
            if family is not None else
            ("ACC-" if name.startswith("aRACE_") else "DDM-")
            + MODEL_LABELS[name.rsplit("_", 1)[-1]]
            for name in display_models
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=28 if family is None else 0,
            ha="right" if family is None else "center",
            fontsize=9 if family is None else 10,
        )
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)])
        ax.set_ylabel("Mean softmax probability", fontsize=10)
        ax.grid(axis="y", alpha=0.20)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, float(value) + 0.014,
                f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=9,
            )
        ax.text(
            0.99, 0.985, f"n = {len(logits)}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#5C6673",
        )
        if family is None:
            ax.axvline(3.5, color="#AAB1BC", lw=1, ls="--")

    draw(ax_acc, ACC_4, "ACC")
    draw(ax_ddm, DDM_4, "DDM")
    draw(ax_all, DDM_4 + ACC_4, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=300, bbox_inches="tight",
        pad_inches=0.22, facecolor="white",
    )
    plt.close(fig)
    print(f"[SAVED] {output_path}")

def save_single_figure(img, title, out_path, figsize=(8.2, 4.8), dpi=300):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.92, left=0.02, right=0.98, bottom=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def save_stacked_figure(img_top, img_bottom, title, top_label, bottom_label,
                        out_path, figsize=(8.4, 10.5), dpi=300):
    fig, axes = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.16},
    )
    for ax, img, label in zip(axes, (img_top, img_bottom), (top_label, bottom_label)):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=6)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.992)
    fig.subplots_adjust(top=0.95, left=0.02, right=0.98, bottom=0.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("compare_regen", "compare_noregen"),
                        default="compare_regen",
                        help="Use this pipeline for BOTH simulated logits and real-data figures.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output_dir.expanduser().resolve() if args.output_dir else (
        root / "outputs" / "plots" / "mean_prob_paper_figs" / args.pipeline
    )
    food_simulated = render_simulated_panel(root, "food_equal", args.pipeline)
    kraj_simulated = render_simulated_panel(root, "krajbich2010", args.pipeline)

    save_single_figure(
        food_simulated, "Food Equal (Simulated)",
        out / "figure_1_food_equal_simulated_mean_prob.png",
        figsize=(10.2, 4.9),
    )
    render_real_figure(
        root, "food_equal", args.pipeline,
        out / "figure_2_food_equal_real_mean_prob.png",
    )
    save_single_figure(
        kraj_simulated, "Krajbich et al. (2010) (Simulated)",
        out / "figure_3_krajbich2010_simulated_mean_prob.png",
        figsize=(10.2, 4.9),
    )
    render_real_figure(
        root, "krajbich2010", args.pipeline,
        out / "figure_4_krajbich2010_real_mean_prob.png",
    )

    # Figure 3 used to denote Krajbich real data. Remove only that obsolete
    # composition output after its successor (Figure 4) has been written.
    outdated = out / "figure_3_krajbich2010_real_mean_prob.png"
    if outdated.is_file():
        outdated.unlink()
        print(f"[REMOVED obsolete output] {outdated}")

    print(f"\nComplete: 4 PNGs in {out}")


if __name__ == "__main__":
    main()
