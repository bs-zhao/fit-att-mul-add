from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIT_DIR = ROOT / "run" / "fit_nn" / "food_equal"
sys.path.insert(0, str(FIT_DIR))
from tools.model_info import model_infos


# s6_decode_maxT.py currently compares these four DDM variants.
MODEL_NAMES = [
    "aDDM_1",
    "aDDM_2",
    "aDDM_t",
    "aDDM_g",
]



MODEL_NAMES = [
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]

MODEL_NAMES = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]


SAVE_NAME = "_".join(MODEL_NAMES)

CSV_PATH = (
    ROOT
    / "outputs"
    / "food_equal"
    / "compare_regen"
    / "real_data"
    / "trial"
    / "s6_decode_trial"
    / f"{SAVE_NAME}.csv"
)

FIG_DIR = ROOT / "outputs" / "food_equal" / "plots"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_NAMES = [model_infos[name]["disp_mname"] for name in MODEL_NAMES]


if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Model-comparison output not found:\n{CSV_PATH}\n"
        "Run run/fit_nn/food_equal/compare_regen/s6_decode_maxT.py first."
    )


df = pd.read_csv(CSV_PATH)
missing = [name for name in MODEL_NAMES if name not in df.columns]
if missing:
    raise ValueError(f"Missing model columns in {CSV_PATH}: {missing}")

scores = df[MODEL_NAMES].to_numpy(dtype=float)
winner_idx = np.argmax(scores, axis=1)
winner_score = np.max(scores, axis=1)

# Sort participants first by winning model, then by confidence within model.
order = np.lexsort((-winner_score, winner_idx))
scores_sorted = scores[order]
winner_idx_sorted = winner_idx[order]

print(f"Loaded: {CSV_PATH}")
print(f"N participants: {len(df)}")
print("\nMean classifier output:")
for name, display, value in zip(MODEL_NAMES, DISPLAY_NAMES, scores.mean(axis=0)):
    print(f"  {display:12s} ({name}): {value:.4f}")

print("\nWinning model counts:")
for i, (name, display) in enumerate(zip(MODEL_NAMES, DISPLAY_NAMES)):
    n = int(np.sum(winner_idx == i))
    print(f"  {display:12s} ({name}): {n}/{len(df)} = {n / len(df):.3f}")


# -----------------------------------------------------------------------------
# Figure 1: participant-level classifier outputs.
# Each row is one participant; participants are grouped by their winning model.
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.16 * len(df))))
im = ax.imshow(scores_sorted, aspect="auto", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(MODEL_NAMES)))
ax.set_xticklabels(DISPLAY_NAMES, rotation=35, ha="right")
ax.set_yticks(np.arange(len(df)))
ax.set_yticklabels(np.arange(1, len(df) + 1))
ax.set_xlabel("Candidate model")
ax.set_ylabel("Participant (sorted)")
ax.set_title("Neural model-comparison output by participant")

# Draw separators between groups with different winning models.
for i in range(1, len(df)):
    if winner_idx_sorted[i] != winner_idx_sorted[i - 1]:
        ax.axhline(i - 0.5, linewidth=1, color="black")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Classifier output")

fig.tight_layout()
file_subject = FIG_DIR / "model_comparison_subjects.png"
fig.savefig(file_subject, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {file_subject}")


# -----------------------------------------------------------------------------
# Figure 2: proportion of participants for which each model has the largest
# classifier output.
# -----------------------------------------------------------------------------
counts = np.bincount(winner_idx, minlength=len(MODEL_NAMES))
proportions = counts / len(df)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(DISPLAY_NAMES, proportions)
ax.set_ylim(0, 1)
ax.set_ylabel("Proportion of participants")
ax.set_xlabel("Winning model")
ax.set_title("Model-comparison winners")
ax.tick_params(axis="x", rotation=35)

for bar, count, prop in zip(bars, counts, proportions):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{count}\n({prop:.2f})",
        ha="center",
        va="bottom",
        fontsize=10,
    )

fig.tight_layout()
file_winner = FIG_DIR / "model_comparison_winners.png"
fig.savefig(file_winner, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {file_winner}")
