"""Unit tests for Bayes-consistency algebra; no GPU/BayesFlow required."""
import sys
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algebra import conditional_probabilities, summarize, write_union_csv

class TestConditioning(unittest.TestCase):
    def test_joint_conditional_recovers_exact_four_way(self):
        # Two families with different prior masses but the same within-family vectors.
        q4 = np.array([[.1, .2, .3, .4], [.7, .1, .1, .1]])
        p8 = np.concatenate([q4 * np.array([[.2],[.9]]),
                              np.ones_like(q4)*np.array([[.8],[.1]])/4],axis=1)
        names = ["d1","d2","d3","d4","r1","r2","r3","r4"]
        out,mass=conditional_probabilities(p8,names[:4],names)
        np.testing.assert_allclose(out,q4)
        np.testing.assert_allclose(mass,[.2,.9])

    def test_zero_mass_rejected(self):
        with self.assertRaises(ValueError):
            conditional_probabilities([[0,0,1]], ["x","y"], ["x","y","z"])

    def test_heterogeneous_family_columns_are_preserved(self):
        # Reproduces the failure after 256 DDM rows from the uploaded zip.
        rows = [
            {"family":"ddm","subj":"s1","four__aDDM_1":.7},
            {"family":"race","subj":"s1","four__aRACE_1":.8},
        ]
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "joint.csv"
            write_union_csv(output, rows)
            with output.open(newline="",encoding="utf-8") as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0]["four__aDDM_1"],"0.7")
            self.assertEqual(saved[1]["four__aRACE_1"],"0.8")
            self.assertEqual(saved[0]["four__aRACE_1"],"")
            self.assertEqual(saved[1]["four__aDDM_1"],"")

if __name__ == "__main__":
    unittest.main()
