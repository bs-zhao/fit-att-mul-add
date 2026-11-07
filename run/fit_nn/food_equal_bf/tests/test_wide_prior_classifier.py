import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.common import model_infos, parser, directory
from experiments.wide_prior_classifier import (
    FAMILIES, OUTPUT, GAMMA_PRIOR, THETA_PRIOR, apply_wide_priors,
    expected_spec, verify_or_write_spec
)


class WidePriorTest(unittest.TestCase):
    def test_correct_parameters_only(self):
        old = {name: (list(v["free_pranges"]), list(v["free_pranges_ml"]))
               for name, v in model_infos.items()}
        try:
            apply_wide_priors()
            for names in FAMILIES.values():
                self.assertEqual(model_infos[names[2]]["free_pranges"][2], THETA_PRIOR)
                self.assertEqual(model_infos[names[3]]["free_pranges"][2], GAMMA_PRIOR)
                for name in names[:2]:
                    self.assertEqual(model_infos[name]["free_pranges"], old[name][0])
                for name in names[2:]:
                    self.assertEqual(model_infos[name]["free_pranges"][:2], old[name][0][:2])
                self.assertEqual(model_infos[names[2]]["free_pranges_ml"][2], THETA_PRIOR)
                self.assertEqual(model_infos[names[3]]["free_pranges_ml"][2], GAMMA_PRIOR)
        finally:
            for name, (prior, ml_prior) in old.items():
                model_infos[name]["free_pranges"] = prior
                model_infos[name]["free_pranges_ml"] = ml_prior

    def test_output_isolated_and_metadata_checked(self):
        a = parser("test", classifier=True).parse_args(
            ["--models", *FAMILIES["race"], "--output", str(OUTPUT)])
        result_dir = directory(a, classifier=True)
        self.assertIn("prior_sensitivity", str(result_dir))
        old = {name: (list(v["free_pranges"]), list(v["free_pranges_ml"]))
               for name, v in model_infos.items()}
        try:
            apply_wide_priors()
            spec = expected_spec(FAMILIES["race"], "formal")
            with tempfile.TemporaryDirectory() as root:
                folder = Path(root)
                verify_or_write_spec(folder, spec, "s4")
                self.assertTrue((folder/"prior_spec.json").exists())
                with self.assertRaises(FileNotFoundError):
                    verify_or_write_spec(folder, spec, "s5")
                with self.assertRaises(ValueError):
                    verify_or_write_spec(folder, dict(spec, theta=[0, .9]), "s4")
        finally:
            for name, (prior, ml_prior) in old.items():
                model_infos[name]["free_pranges"] = prior
                model_infos[name]["free_pranges_ml"] = ml_prior


if __name__ == "__main__":
    unittest.main()
