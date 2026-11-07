import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from experiments.race_add_cross_architecture import DDM, RACE, classify, paired_summary


class FakeEstimator:
    def estimate(self, conditions, model_prior):
        assert np.allclose(model_prior, [.25]*4)
        v = np.array([[0, -1000, 1000, 500], [0, -1000, -500, 900]], dtype=float)
        z = v - v.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return {"model_probs": p, "log_bayes_factors": v}


class CrossArchitectureTests(unittest.TestCase):
    def test_stable_log_bf_under_saturated_probabilities(self):
        raw = {"trials": np.zeros((2, 3, 17))}
        for names in [DDM, RACE]:
            probs, logbf = classify(FakeEstimator(), raw, names)
            np.testing.assert_allclose(logbf, [500, -1400])
            np.testing.assert_allclose(probs.sum(1), 1)

    def test_mul_and_add_generating_model_labels(self):
        df = pd.DataFrame({"DDM_logBF_Mul_Add": [-1., 2., 3.],
                           "RACE_logBF_Mul_Add": [-2., 1., 4.]})
        for generator in ["aRACE_g", "aRACE_t"]:
            result = paired_summary(df, 30, 12, generator=generator)
            self.assertEqual(result["generator"], generator)
            self.assertEqual(result["n"], 3)
            self.assertEqual(result["n_DDM_stronger"], 2)

    def test_paired_delta_sign(self):
        df = pd.DataFrame({"DDM_logBF_Mul_Add": [3., 4., 5., 6.],
                           "RACE_logBF_Mul_Add": [-1., 0., 1., 2.]})
        summary = paired_summary(df, 100, 11)
        self.assertAlmostEqual(summary["mean_delta"], 4)
        self.assertEqual(summary["n_DDM_stronger"], 4)
        self.assertEqual(summary["DDM"]["Mul_wins_vs_Add"], 4)
        self.assertEqual(summary["RACE"]["Add_wins_vs_Mul"], 1)


if __name__ == "__main__":
    unittest.main()
