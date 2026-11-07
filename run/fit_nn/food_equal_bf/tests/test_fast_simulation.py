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

    def test_nonbinary_gaze_codes_match_legacy_summary_and_features(self):
        """Off-target gaze codes remain in simulations, but not gaze summaries."""
        from _shared.data import funcs
        cases = [
            ([0, 2, 0, 1], [100, 25, 50, 300]),  # off-target splits same-eye runs
            ([-1, 1, 3, 1, 0], [250, 60, 80, 100, 240]),
            ([2, 3, -1], [100, 250, 400]),          # no valid AOI
            ([0, 2, 1, 2, 0], [0, 30, 0, 120, 50]),
        ]
        for pp, dd in cases:
            with self.subTest(pos=pp, dur=dd):
                p, d = np.asarray(pp, np.int64), np.asarray(dd, np.int64)
                legacy = funcs.get_summary(funcs.arr2tp(p,d),.001)
                np.testing.assert_allclose(fast_fixation_summary(p,d),legacy,rtol=0,atol=1e-12)
                fast = feature([4,8],0,max(.001,d.sum()*.001),p,d)
                slow = feature([4,8],0,max(.001,d.sum()*.001),p,d,legacy=True)
                np.testing.assert_array_equal(fast,slow)

        rng = np.random.default_rng(20261107)
        for _ in range(250):
            n = int(rng.integers(1,30))
            p = rng.choice(np.array([-1,0,1,2,3],np.int64),size=n)
            d = rng.integers(0,101,size=n,dtype=np.int64)
            a = feature([3,6],1,max(.001,d.sum()*.001),p,d)
            b = feature([3,6],1,max(.001,d.sum()*.001),p,d,legacy=True)
            np.testing.assert_array_equal(a,b)

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
