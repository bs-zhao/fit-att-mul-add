"""Deterministic safety tests for flexible eight-model shared-prior sweeps."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _shared.data import ROOT, LEGACY, MODEL_NAMES, model_infos, prior_spec, _module
from experiments import shared_prior_sweep as sweep
from experiments.wide_prior_classifier import verify_or_write_spec


class SharedPriorSweepTests(unittest.TestCase):
    def setUp(self):
        self.initial_bounds = {
            name: {key: list(info[key]) for key in ("free_pranges", "free_pranges_ml")}
            for name, info in model_infos.items()
        }
        self.initial_profile = (sweep.D_PRIOR, sweep.A_PRIOR)

    def tearDown(self):
        for name, bounds in self.initial_bounds.items():
            for key, values in bounds.items():
                model_infos[name][key] = values
        sweep.configure_priors(*self.initial_profile[0], *self.initial_profile[1])

    def test_shared_priors_across_all_eight_models(self):
        original = _module("legacy_original_model_infos_for_sweep_test",
                           LEGACY / "tools/model_info.py").model_infos
        for d, a in [((0.1, 5.0), (1.0, 5.0)),
                     ((0.1, 6.0), (0.5, 5.0)),
                     ((0.1, 5.0), (0.5, 4.0))]:
            with self.subTest(d=d, a=a):
                sweep.configure_priors(*d, *a)
                sweep.apply_shared_da_priors()
                self.assertEqual(len(MODEL_NAMES), 8)
                for name, info in model_infos.items():
                    for pkey, rkey in (("free_pnames", "free_pranges"),
                                       ("free_pnames_ml", "free_pranges_ml")):
                        found = dict(zip(info[pkey], info[rkey]))
                        self.assertEqual(found["d"], d)
                        self.assertEqual(found["a"], a)
                        if name.endswith("_t"):
                            self.assertEqual(found["theta"], (0.0, 1.0))
                        if name.endswith("_g"):
                            self.assertEqual(found["gamma"], (0.0, 5.0))
                    self.assertEqual(info["fixed_pnames"], original[name]["fixed_pnames"])
                    self.assertEqual(info["fixed_pvalues"], original[name]["fixed_pvalues"])

    def test_separate_output_roots_for_profiles_and_previous_experiments(self):
        legacy_results = [
            ROOT / "outputs/food_equal_bf",
            ROOT / "outputs/food_equal_bf/prior_sensitivity/"
                   "shared_d0p1to5_a0p5to5_theta0to1_gamma0to5",
            ROOT / "outputs/food_equal_bf/prior_sensitivity/"
                   "shared_d0p1to8_a0p5to5_theta0to1_gamma0to5",
        ]
        roots = []
        for d, a in [((0.1, 5), (1, 5)),
                     ((0.1, 6), (0.5, 5)),
                     ((0.1, 5), (0.5, 4))]:
            sweep.configure_priors(*d, *a)
            roots.append(sweep.OUTPUT)
            self.assertTrue(sweep.OUTPUT.name.startswith("sweep_v1_shared_"))
            self.assertFalse(any(root == sweep.OUTPUT for root in legacy_results))
            paths = [sweep.classifier_folder(f, "formal") for f in ("ddm", "race", "all")]
            self.assertEqual(len(set(paths)), 3)
            self.assertTrue(all(x.is_relative_to(sweep.OUTPUT) for x in paths))
        self.assertEqual(len(set(roots)), 3)

    def test_reject_invalid_bounds(self):
        values = [
            (0, 5, 1, 5), (5, 5, 1, 5), (6, 5, 1, 5),
            (0.1, 5, 0, 5), (0.1, 5, 5, 5),
            (0.1, math.inf, 1, 5), (math.nan, 5, 1, 5),
        ]
        for bounds in values:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    sweep.configure_priors(*bounds)

    def test_checkpoint_provenance_and_prior_conflict(self):
        sweep.configure_priors(0.1, 5, 1, 5)
        sweep.apply_shared_da_priors()
        spec = sweep.expected_spec(sweep.FAMILIES["all"], "formal")
        self.assertEqual(spec["d"], [0.1, 5.0])
        self.assertEqual(spec["a"], [1.0, 5.0])
        self.assertEqual(spec["theta"], [0.0, 1.0])
        self.assertEqual(spec["gamma"], [0.0, 5.0])
        self.assertEqual(spec["free_priors"], prior_spec(MODEL_NAMES)["free_priors"])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            verify_or_write_spec(folder, spec, "s4")
            self.assertEqual(json.loads((folder / "prior_spec.json").read_text()), spec)
            with self.assertRaises(ValueError):
                verify_or_write_spec(folder, dict(spec, a=[0.5, 5]), "s4")
            with self.assertRaises(FileNotFoundError):
                verify_or_write_spec(folder, spec, "s5")


if __name__ == "__main__":
    unittest.main()
