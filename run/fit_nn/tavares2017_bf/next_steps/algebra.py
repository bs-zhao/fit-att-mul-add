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



def write_union_csv(path, rows):
    """Save heterogeneous dict records atomically with the ordered union of keys."""
    import csv
    import os
    import tempfile
    from pathlib import Path
    path = Path(path)
    if not rows:
        raise ValueError("Cannot save empty CSV")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                delete=False) as handle:
            tmp = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
