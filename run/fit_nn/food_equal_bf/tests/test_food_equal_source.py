"""Food Equal source-of-truth invariants; no BayesFlow training required."""
import sys
import unittest
import tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from _shared.data import (LEGACY, MODEL_NAMES, model_infos, feature, funcs,
                          MAX_RT, MAX_LEN_RT, MAX_LEN_NF, SCALE, fast_fixation_summary,
                          _module, ATTENTION_PRIORS, PRIOR_PROFILE, prior_spec)


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
                         [(0.1,8),(0.5,5),(0.0,5.0)])
        self.assertEqual(model_infos["aDDM_g"]["free_pranges"],
                         [(0.1,5),(1,5),(0.0,5.0)])
        self.assertEqual(model_infos["aRACE_t"]["free_pranges"],
                         [(0.1,8),(0.5,5),(0.0,1.0)])
        self.assertEqual(model_infos["aDDM_t"]["free_pranges"],
                         [(0.1,5),(1,5),(0.0,1.0)])

    def test_all_wide_attention_priors_and_legacy_file_unchanged(self):
        legacy = _module("food_original_prior_audit", LEGACY/"tools/model_info.py").model_infos
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                now, original = model_infos[name], legacy[name]
                self.assertEqual(now["free_pnames"], original["free_pnames"])
                self.assertEqual(now["free_pnames_ml"], original["free_pnames_ml"])
                self.assertEqual(now["fixed_pvalues"], original["fixed_pvalues"])
                self.assertEqual(now["fixed_pnames"], original["fixed_pnames"])
                self.assertEqual(now["free_pranges"][:2], original["free_pranges"][:2])
                self.assertEqual(now["free_pranges_ml"][:2], original["free_pranges_ml"][:2])
                if name.endswith("_t"):
                    self.assertEqual(now["free_pranges"][2], ATTENTION_PRIORS["theta"])
                    self.assertEqual(now["free_pranges_ml"][2], ATTENTION_PRIORS["theta"])
                elif name.endswith("_g"):
                    self.assertEqual(now["free_pranges"][2], ATTENTION_PRIORS["gamma"])
                    self.assertEqual(now["free_pranges_ml"][2], ATTENTION_PRIORS["gamma"])
                else:
                    self.assertEqual(now["free_pranges"], original["free_pranges"])
                    self.assertEqual(now["free_pranges_ml"], original["free_pranges_ml"])
        self.assertEqual(legacy["aRACE_g"]["free_pranges"][2], (0.1, 1))
        self.assertEqual(legacy["aDDM_g"]["free_pranges"][2], (0.0, 1))
        self.assertEqual(prior_spec(["aRACE_g"])["profile"], PRIOR_PROFILE)
        self.assertEqual(prior_spec(["aRACE_g"])["free_priors"]["aRACE_g"]["gamma"], [0.0,5.0])

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

    def test_real_data_skips_and_audits_invalid_trials(self):
        """No response/zero-RT trials are excluded, not recoded."""
        from _shared.data import Experiment
        rows = []
        def record(subj, choice=1, durations="[200, 300]", positions="[0, 1]"):
            rows.append(dict(subj=subj, v0=4, v1=7, response=choice,
                             arr_ml3_left=positions, arr_ml3_time=durations))
        for i in range(8):
            record(1, choice=i % 2)
        record(1, choice=-1)
        record(1, choice=2)
        record(1, choice=0, durations="[0, 0]")
        record(1, choice=0, durations="[250, -10]")
        record(1, choice=1, durations="[7500, 7500]")
        for i in range(5):
            record(2, choice=i % 2)
        record(2, choice=-1)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'trial_eye.csv'
            pd.DataFrame(rows).to_csv(file, index=False)
            ex = Experiment(file)
            ids, raw, audit = ex.real()
        self.assertEqual(ids, [1])
        self.assertEqual(raw['trials'].shape[0], 1)
        self.assertEqual(int(raw['summary_mask'][0].sum()), 8)
        self.assertEqual(audit[0]['input_trials'], 13)
        self.assertEqual(audit[0]['retained_trials'], 8)
        self.assertEqual(audit[0]['excluded_trials'], 5)
        self.assertEqual(audit[0]['invalid_choice'], 2)
        self.assertEqual(audit[0]['invalid_rt'], 1)
        self.assertEqual(audit[0]['invalid_fixation'], 1)
        self.assertEqual(audit[0]['maxT_excluded'], 1)
        self.assertTrue(audit[0]['included'])
        self.assertEqual(audit[1]['retained_trials'], 5)
        self.assertFalse(audit[1]['included'])
        self.assertEqual(len(ex.real_exclusions), 6)
        self.assertEqual({v['reason'] for v in ex.real_exclusions},
                         {'invalid_choice','invalid_rt','invalid_fixation','maxT_excluded'})

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
