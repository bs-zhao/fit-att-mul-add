"""Regression coverage for normalization and the model/range contract across datasets."""

import csv
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PIPELINES = Path(__file__).resolve().parents[1]


class UnifiedG2Tests(unittest.TestCase):
    def test_three_adapters_use_the_same_models_and_one_value_scale(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "trial_eye.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["subj", "v0", "v1", "response", "arr_ml3_left", "arr_ml3_time"])
                for subject, values in [(1, (-2, 4)), (2, (10, 5))]:
                    writer.writerows(
                        (subject, *values, trial % 2, "[0, 1]", "[250, 350]")
                        for trial in range(6)
                    )
            expected = None
            for dataset in ("food_equal", "krajbich2010", "tavares2017"):
                with self.subTest(dataset=dataset):
                    data = run_path(str(PIPELINES / f"{dataset}_bf/_shared/data.py"))
                    experiment = data["Experiment"](path, unified=True)
                    _, real, _ = experiment.real()
                    self.assertEqual(experiment.value_min, -2.0)
                    self.assertEqual(experiment.value_max, 10.0)
                    values = [[-0.2, 0.4], [1.0, 0.5]]
                    np.testing.assert_allclose(real["trials"][:, 0, :2], values)
                    np.testing.assert_allclose(
                        np.stack([trials[0][0] for trials in experiment._designs]), values
                    )
                    specification = data["unified_g2"].checkpoint_spec(
                        experiment, list(experiment.models)
                    )
                    self.assertEqual(len(experiment.models), 8)
                    for priors in specification["free_priors"].values():
                        self.assertEqual(priors["d"], [1.0, 50.0])
                        self.assertEqual(priors["a"], [0.5, 5.0])
                        if "theta" in priors:
                            self.assertEqual(priors["theta"], [0.0, 1.0])
                        if "gamma" in priors:
                            self.assertEqual(priors["gamma"], [0.0, 2.0])
                    for fixed in specification["fixed_parameters"].values():
                        self.assertEqual(fixed["ndt"], 0)
                        self.assertEqual(fixed["s"], 1)
                    if expected is None:
                        expected = specification
                    self.assertEqual(specification, expected)


if __name__ == "__main__":
    unittest.main()
