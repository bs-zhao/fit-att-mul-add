"""Food Equal source-of-truth invariants; no BayesFlow training required."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from _shared.data import (LEGACY, MODEL_NAMES, model_infos, feature, funcs,
                          MAX_RT, MAX_LEN_RT, MAX_LEN_NF, SCALE, fast_fixation_summary)


class FoodEqualSourceTests(unittest.TestCase):
    def test_model_catalog_is_original_food_equal(self):
        original = (LEGACY / "tools" / "model_info.py").read_text(encoding="utf-8")
        self.assertIn('"aDDM_1"', original)
        self.assertEqual(MODEL_NAMES, [
            "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
            "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
        ])
        self.assertEqual(model_infos["aDDM_1"]["free_pranges"], [(0.1,5),(1,5)])
        self.assertEqual(model_infos["aRACE_g"]["free_pranges"],
                         [(0.1,8),(0.5,5),(0.1,1)])
        self.assertEqual(model_infos["aDDM_g"]["free_pranges"],
                         [(0.1,5),(1,5),(0.0,1)])

    def test_food_equal_rt_and_fixation_cutoffs(self):
        self.assertEqual(MAX_RT, 14.5)
        self.assertEqual(MAX_LEN_RT, 15000)
        self.assertEqual(MAX_LEN_NF, 128)
        self.assertEqual(float(SCALE[4]), 15.)
        self.assertEqual(float(SCALE[9]), 128.)
        self.assertIsNotNone(feature([3, 8], 0, 1., [0, 1], [7000, 7997]))
        self.assertIsNone(feature([3, 8], 0, 1., [0, 1], [7000, 7998]))
        self.assertIsNone(feature([3, 8], 0, 1., np.r_[np.zeros(128, dtype=int)],
                                  np.full(128, 100)))

    def test_food_equal_legacy_fixation_summary(self):
        p = np.array([0, 1, 1, 0], dtype=np.int64)
        d = np.array([130, 200, 110, 250], dtype=np.int64)
        observed = fast_fixation_summary(p, d)
        source = funcs.get_summary(funcs.arr2tp(p, d), .001)
        np.testing.assert_array_equal(np.asarray(observed, np.float32),
                                      np.asarray(source, np.float32))

    def test_unchanged_source_parameters_are_not_mutated(self):
        info = model_infos["aRACE_t"]
        params = funcs.make_params(info, np.random.default_rng(131))
        for name,(lo,hi) in zip(info["free_pnames"],info["free_pranges"]):
            self.assertGreaterEqual(params[name],lo)
            self.assertLess(params[name],hi)
        for k,v in zip(info["fixed_pnames"],info["fixed_pvalues"]):
            self.assertEqual(params[k],v)


if __name__ == "__main__":
    unittest.main()
