"""Fast feature extraction and reconstructing rcd=0 gaze sequences."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from _shared.data import feature, fast_fixation_summary, reconstruct_fixations


class FastSimulationTests(unittest.TestCase):
    def test_handcrafted_fixation_features(self):
        data = [
            ([0], [100]),
            ([0, 0, 1, 1, 0], [0, 10, 0, 70, 6]),
            ([1, 1, 1], [100, 0, 300]),
            ([0, 1, 0, 1], [250, 350, 300, 400]),
        ]
        for pp, dd in data:
            p, d = np.asarray(pp, dtype=np.int64), np.asarray(dd, dtype=np.int64)
            np.testing.assert_array_equal(
                feature([3, 8], 0, max(.001, d.sum() * .001), p, d),
                feature([3, 8], 0, max(.001, d.sum() * .001), p, d, legacy=True)
            )

    def test_randomized_features_match_legacy_exactly(self):
        rng = np.random.default_rng(17)
        for _ in range(300):
            count = int(rng.integers(1, 50))
            pos = rng.integers(0, 2, count, dtype=np.int64)
            durations = rng.integers(0, 100, count, dtype=np.int64)
            x = feature([5, 6], 1, .5, pos, durations)
            ref = feature([5, 6], 1, .5, pos, durations, legacy=True)
            np.testing.assert_array_equal(x, ref)

    def test_final_fixation_extension_and_boundaries(self):
        pos = np.array([0, 1, 0], np.int64)
        dur = np.array([300, 500, 400], np.int64)
        examples = [
            (.001, [0], [1]),
            (.300, [0], [300]),
            (.301, [0, 1], [300, 1]),
            (.800, [0, 1], [300, 500]),
            (.950, [0, 1, 0], [300, 500, 150]),
            (1.500, [0, 1, 0], [300, 500, 700]),
        ]
        for rt, pp, dd in examples:
            p, d = reconstruct_fixations(pos, dur, rt)
            np.testing.assert_array_equal(p, pp)
            np.testing.assert_array_equal(d, dd)

    def test_zero_duration_segment(self):
        p, d = reconstruct_fixations(
            np.array([0, 1, 1]), np.array([0, 200, 10]), .001)
        np.testing.assert_array_equal(p, [0, 1])
        np.testing.assert_array_equal(d, [0, 1])


if __name__ == "__main__":
    unittest.main()
