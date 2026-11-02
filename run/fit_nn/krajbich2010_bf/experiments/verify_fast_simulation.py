"""Verify stochastic/feature parity of fast simulation vs original S4 path.

For each of four selected models: reset identical NumPy and libc RNG states,
simulate the same empirical subjects with rcd=1 / expanded features (legacy)
and rcd=0 / reconstructed fixations / direct features (fast), and assert
equality. Must PASS before launching expensive optimized S4 training.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse
import ctypes
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from _shared.common import ROOT, environment
from _shared.data import Experiment, feature, reconstruct_fixations
from experiments.wide_prior_classifier import FAMILIES, apply_wide_priors


def seed_c(seed):
    ctypes.CDLL(None).srand(seed % (2**32))


def feature_parity():
    rng = np.random.default_rng(314159)
    cases = [
        ([0], [1]),
        ([0, 1, 0], [100, 200, 300]),
        ([0, 0, 1, 1, 0], [0, 10, 0, 70, 6]),
        ([1, 1, 0, 0, 1], [15, 0, 11, 3, 2]),
    ]
    for _ in range(500):
        n = int(rng.integers(1, 20))
        cases.append((rng.integers(0, 2, n),
                      rng.integers(0, 3500, n)))
    for p, d in cases:
        p, d = np.asarray(p, np.int64), np.asarray(d, np.int64)
        if d.sum() >= 28287:
            continue
        fast = feature([3., 6.], 1, max(.001, d.sum() * .001), p, d)
        slow = feature([3., 6.], 1, max(.001, d.sum() * .001),
                       p, d, legacy=True)
        np.testing.assert_array_equal(fast, slow,
                                      err_msg=f"feature mismatch p={p}, d={d}")
    print(f"[PASS] Exact float32 feature parity on {len(cases)} cases", flush=True)


def model_parity(ex, model, n_subjects, base_seed):
    def run(legacy):
        seed_c(base_seed)
        rng = np.random.default_rng(base_seed + 7)
        stats = dict(trials=0, nonhits=0, maxT_excluded=0)
        t0 = time.perf_counter()
        results = [ex.simulate(model, rng, stats, legacy=legacy)
                   for _ in range(n_subjects)]
        return results, stats, time.perf_counter() - t0

    baseline, old_stats, old_secs = run(True)
    optimized, new_stats, new_secs = run(False)
    if old_stats != new_stats:
        raise AssertionError(f"{model} rejection counts changed: {old_stats} vs {new_stats}")
    n_rows = 0
    for (old_p, old_rows), (new_p, new_rows) in zip(baseline, optimized):
        if old_p != new_p or len(old_rows) != len(new_rows):
            raise AssertionError(f"{model}: sampled parameters or retained trials differ")
        for old, new in zip(old_rows, new_rows):
            np.testing.assert_array_equal(old, new,
                                          err_msg=f"{model}: trial feature mismatch")
            n_rows += 1
    print(f"[PASS] {model}: {n_subjects} subjects, {n_rows} retained trials, "
          f"legacy {old_secs:.2f}s -> fast {new_secs:.2f}s "
          f"({old_secs / new_secs:.2f}x for simulation only)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=FAMILIES, required=True)
    ap.add_argument("--n-subjects", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261102)
    args = ap.parse_args()
    if args.n_subjects < 1:
        ap.error("--n-subjects must be positive")
    environment(args.seed)
    apply_wide_priors()
    feature_parity()
    ex = Experiment(ROOT / "data/krajbich2010/trial_eye.csv")
    for i, name in enumerate(FAMILIES[args.family]):
        model_parity(ex, name, args.n_subjects, args.seed + 301 * (i + 1))
    print("[ALL PASS] Optimized simulation is equivalent to the original.", flush=True)


if __name__ == "__main__":
    main()
