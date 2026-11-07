"""Unit tests for current-prior model comparison figures and summaries."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from experiments.report_model_comparison import evidence_groups, pairwise_matrix


class ReportTests(unittest.TestCase):
    def test_bayes_factor_strength_bins(self):
        np.testing.assert_array_equal(
            evidence_groups(np.array([-4.0, -1.0, .5, 4.0])),
            np.array([1, 1, 1, 1]))

    def test_pairwise_evidence_directions(self):
        names = ["aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g"]
        offsets = [0., 1., 2., 3.]
        data = [dict(numerator=a, denominator=b,
                     log_bayes_factor=offsets[i] - offsets[j])
                for i,a in enumerate(names) for j,b in enumerate(names) if i!=j]
        mean, pct = pairwise_matrix(pd.DataFrame(data), names)
        self.assertEqual(mean[2,3], -1)
        self.assertEqual(mean[3,2], 1)
        self.assertEqual(pct[2,3], 0)
        self.assertEqual(pct[3,2], 100)


if __name__ == "__main__":
    unittest.main()
