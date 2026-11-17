"""Shared, non-neural posterior conditioning and TV metrics."""
import numpy as np

def conditional_probabilities(eight_probs, family_names, all_names):
    arr = np.asarray(eight_probs, dtype=np.float64)
    indices = [all_names.index(m) for m in family_names]
    if arr.ndim != 2 or arr.shape[1] != len(all_names):
        raise ValueError("Wrong joint probability shape")
    joint = arr[:, indices]
    mass = joint.sum(axis=1)
    if np.any(mass <= 1e-12):
        raise ValueError("Joint family mass too small for reliable conditioning")
    return joint / mass[:, None], mass


def summarize(rows):
    if not rows:
        raise ValueError("Empty comparison")
    return {
        "n": len(rows),
        "mean_tv": float(np.mean([r["tv"] for r in rows])),
        "median_tv": float(np.median([r["tv"] for r in rows])),
        "q90_tv": float(np.quantile([r["tv"] for r in rows], .9)),
        "max_tv": float(np.max([r["tv"] for r in rows])),
        "winner_agreement": float(np.mean([r["winner_agreement"] for r in rows])),
        "mean_joint_family_mass": float(np.mean([r["joint_family_mass"] for r in rows])),
    }


