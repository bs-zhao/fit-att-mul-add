"""Compose three publication-style mean_prob figures for one chosen pipeline.

All simulated panels are calculated from the *selected pipeline's* aRACE_g
test-data classifier logits, separately within ACC_4 and DDM_4. Real-data
panels are assembled from that pipeline's existing mean_prob PNGs.

Example:
    python run/plot/compose_mean_prob_paper_figs.py --pipeline compare_regen
    python run/plot/compose_mean_prob_paper_figs.py --pipeline compare_noregen

Source real figures: outputs/plots/sbi_9fig_<pipeline>/mean_prob/
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
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title-mask-frac", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.title_mask_frac < 1:
        parser.error("--title-mask-frac must be in [0,1)")
    root = Path(__file__).resolve().parents[2]
    inp = args.input_dir.expanduser().resolve() if args.input_dir else (
        root / "outputs" / "plots" / f"sbi_9fig_{args.pipeline}" / "mean_prob"
    )
    out = args.output_dir.expanduser().resolve() if args.output_dir else (
        root / "outputs" / "plots" / "mean_prob_paper_figs" / args.pipeline
    )
    names = {
        "food4": "figure_4_food_equal_4models_mean_prob.png",
        "food8": "figure_6_food_equal_8models_mean_prob.png",
        "kraj4": "figure_5_krajbich2010_4models_mean_prob.png",
        "kraj8": "figure_7_krajbich2010_8models_mean_prob.png",
    }
    panels = {key: load_panel(inp / name, args.title_mask_frac)
              for key, name in names.items()}
    food_simulated = render_simulated_panel(root, "food_equal", args.pipeline)
    kraj_simulated = render_simulated_panel(root, "krajbich2010", args.pipeline)
    save_stacked_figure(
        food_simulated, kraj_simulated,
        "ACC-Add simulated data",
        "Food Equal", "Krajbich et al. (2010)",
        out / "figure_1_food_equal_simulated_mean_prob.png",
    )
    save_stacked_figure(panels["food4"], panels["food8"],
                        "Food Equal", "4 models", "8 models",
                        out / "figure_2_food_equal_real_mean_prob.png")
    save_stacked_figure(panels["kraj4"], panels["kraj8"],
                        "Krajbich et al. (2010)", "4 models", "8 models",
                        out / "figure_3_krajbich2010_real_mean_prob.png")
    print(f"\nComplete: 3 PNGs in {out}")


if __name__ == "__main__":
    main()
