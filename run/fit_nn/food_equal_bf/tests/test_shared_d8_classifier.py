"""Unit tests for the independent shared d/a Food Equal BF experiment."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.data import ROOT, LEGACY, MODEL_NAMES, model_infos, prior_spec, _module
from _shared.common import directory, parser
from experiments.shared_d8_classifier import (
    EXPERIMENT_ID, OUTPUT, D_PRIOR, A_PRIOR, FAMILIES,
    apply_shared_da_priors, expected_spec, classifier_folder,
)
from experiments.wide_prior_classifier import verify_or_write_spec


class SharedD8BFTests(unittest.TestCase):
    def setUp(self):
        self.before = {name: {
            key: list(info[key]) for key in ("free_pranges", "free_pranges_ml")
        } for name, info in model_infos.items()}

    def tearDown(self):
        for name, original in self.before.items():
            for key, value in original.items():
                model_infos[name][key] = value

    def test_all_eight_models_share_exact_d_a_priors(self):
        apply_shared_da_priors()
        self.assertEqual(len(MODEL_NAMES), 8)
        original = _module("food_equal_unmodified_source_model_infos",
                           LEGACY / "tools/model_info.py").model_infos
        for model in MODEL_NAMES:
            with self.subTest(model=model):
                info = model_infos[model]
                for name_key, bound_key in (
                    ("free_pnames", "free_pranges"),
                    ("free_pnames_ml", "free_pranges_ml"),
                ):
                    bounds = dict(zip(info[name_key], info[bound_key]))
                    self.assertEqual(bounds["d"], D_PRIOR)
                    self.assertEqual(bounds["a"], A_PRIOR)
                    if model.endswith("_t"):
                        self.assertEqual(bounds["theta"], (0.0, 1.0))
                    elif model.endswith("_g"):
                        self.assertEqual(bounds["gamma"], (0.0, 5.0))
                    for parameter, interval in zip(info[name_key], info[bound_key]):
                        if parameter not in ("d", "a", "theta", "gamma"):
                            self.assertEqual(interval,
                                dict(zip(original[model][name_key],
                                         original[model][bound_key]))[parameter])
                self.assertEqual(info["fixed_pnames"], original[model]["fixed_pnames"])
                self.assertEqual(info["fixed_pvalues"], original[model]["fixed_pvalues"])
        # Original legacy source is not changed by BF in-process overrides.
        self.assertEqual(original["aRACE_1"]["free_pranges"][:2],
                         [(0.1, 8), (0.5, 5)])
        self.assertEqual(original["aDDM_1"]["free_pranges"][:2],
                         [(0.1, 5), (1, 5)])

    def test_isolation_from_existing_bf_run_and_priors(self):
        apply_shared_da_priors()
        default_output = ROOT / "outputs/food_equal_bf"
        old_wide = default_output / "prior_sensitivity/gamma_0_5_theta_0_1"
        self.assertNotEqual(OUTPUT, default_output)
        self.assertNotEqual(OUTPUT, old_wide)
        self.assertEqual(OUTPUT.name, EXPERIMENT_ID)
        locations = [classifier_folder(family, "formal") for family in FAMILIES]
        self.assertEqual(len(set(locations)), 3)
        self.assertEqual(locations[-1], OUTPUT / "classifier/formal")
        for family in FAMILIES:
            checkpoint = classifier_folder(family, "formal")
            self.assertTrue(checkpoint.is_relative_to(OUTPUT))
            names = FAMILIES[family]
            spec = expected_spec(names, "formal")
            self.assertEqual(spec["model_names"], names)
            self.assertEqual(spec["d"], [0.1, 8.0])
            self.assertEqual(spec["a"], [0.5, 5.0])
            self.assertEqual(spec["theta"], [0.0, 1.0])
            self.assertEqual(spec["gamma"], [0.0, 5.0])
            self.assertEqual(spec["free_priors"], prior_spec(names)["free_priors"])

    def test_prior_spec_rejects_incompatible_checkpoint(self):
        apply_shared_da_priors()
        expected = expected_spec(FAMILIES["ddm"], "formal")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            verify_or_write_spec(folder, expected, "s4")
            stored = json.loads((folder / "prior_spec.json").read_text())
            self.assertEqual(stored, expected)
            with self.assertRaises(ValueError):
                verify_or_write_spec(folder, dict(expected, d=[0.1, 5.0]), "s4")
            with self.assertRaises(FileNotFoundError):
                verify_or_write_spec(folder, expected, "s5")


if __name__ == "__main__":
    unittest.main()
