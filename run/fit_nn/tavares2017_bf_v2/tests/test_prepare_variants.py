"""Independent, no-BayesFlow tests for all variant input semantics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_variants import convert, PROFILES, value


class TestVariantData(unittest.TestCase):
    def setUp(self):
        self.exp = (
            "parcode,trial,rt,choice,item_left,item_right,valid\n"
            "0,0,1000,-1,0,15,undefined\n"
            "0,1,1000,1,5,10,undefined\n"
        ).encode()
        self.fix = (
            "parcode,trial,fix_item,fix_time\n"
            "0,0,3,100\n0,0,0,100\n0,0,1,300\n0,0,0,100\n0,0,2,400\n"
            "0,1,3,600\n0,1,1,100\n0,1,0,200\n0,1,2,100\n"
        ).encode()

    def test_profiles(self):
        self.assertEqual(len(PROFILES), 6)

    def test_full_relative_and_qc(self):
        rows, audit = convert(self.exp, self.fix, profile="full_relative_qc")
        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["excluded"]["off_item_above_50pct"], 1)
        self.assertEqual(rows[0]["arr_ml3_left"], "[-1, -1, 1, -1, 0]")
        self.assertEqual(sum(__import__("ast").literal_eval(rows[0]["arr_ml3_time"])), 1000)
        self.assertEqual((rows[0]["v0"], rows[0]["v1"]), (0., 1.))
        self.assertEqual(rows[0]["response"], 1)

    def test_active_removes_neutral(self):
        rows, _ = convert(self.exp, self.fix, profile="active_thomas_all")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["arr_ml3_left"], "[1, 0]")
        self.assertEqual(sum(__import__("ast").literal_eval(rows[0]["arr_ml3_time"])), 700)
        self.assertEqual((rows[0]["v0"], rows[0]["v1"]), (1., 10.))

    def test_full_keeps_qc_failures(self):
        rows, _ = convert(self.exp, self.fix, profile="full_thomas_all")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["arr_ml3_left"], "[-1, 1, -1, 0]")

    def test_value(self):
        self.assertAlmostEqual(value(5, "relative"), 2./3.)
        self.assertEqual(value(15, "thomas"), 1.)


if __name__ == "__main__":
    unittest.main()
