"""Compose three paper/response-letter PNG figures from existing mean_prob plots.

Source: outputs/plots/sbi_9fig_compare_regen/mean_prob/
Output: outputs/plots/mean_prob_paper_figs/

No classifier inference is performed; only existing images are rearranged.
Run: python run/plot/compose_mean_prob_paper_figs.py
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


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
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title-mask-frac", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.title_mask_frac < 1:
        parser.error("--title-mask-frac must be in [0,1)")
    root = Path(__file__).resolve().parents[2]
    inp = args.input_dir.expanduser().resolve() if args.input_dir else (
        root / "outputs" / "plots" / "sbi_9fig_compare_regen" / "mean_prob"
    )
    out = args.output_dir.expanduser().resolve() if args.output_dir else (
        root / "outputs" / "plots" / "mean_prob_paper_figs"
    )
    names = {
        "sim": "figure_8_food_equal_simulated_aRACE_g_mean_prob.png",
        "food4": "figure_4_food_equal_4models_mean_prob.png",
        "food8": "figure_6_food_equal_8models_mean_prob.png",
        "kraj4": "figure_5_krajbich2010_4models_mean_prob.png",
        "kraj8": "figure_7_krajbich2010_8models_mean_prob.png",
    }
    panels = {key: load_panel(inp / name, args.title_mask_frac)
              for key, name in names.items()}
    save_single_figure(panels["sim"], "Food Equal (Simulated)",
                       out / "figure_1_food_equal_simulated_mean_prob.png")
    save_stacked_figure(panels["food4"], panels["food8"],
                        "Food Equal", "4 models", "8 models",
                        out / "figure_2_food_equal_real_mean_prob.png")
    save_stacked_figure(panels["kraj4"], panels["kraj8"],
                        "Krajbich et al. (2010)", "4 models", "8 models",
                        out / "figure_3_krajbich2010_real_mean_prob.png")
    print(f"\nComplete: 3 PNGs in {out}")


if __name__ == "__main__":
    main()
