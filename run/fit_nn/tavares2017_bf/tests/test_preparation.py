"""Pure-standard-library preprocessing tests (no BayesFlow/Cython required)."""
import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import convert_data, similarity


class TavaresPreprocessingTests(unittest.TestCase):
    def test_similarity_direction(self):
        self.assertEqual([similarity(x) for x in (0, -5, 10, 15)], [10, 7, 4, 1])

    def test_choice_value_and_off_item_gaze(self):
        trials = (
            "parcode,trial,rt,choice,item_left,item_right,valid\n"
            "0,0,600,-1,0,15,\n"
            "0,1,500,1,-10,5,\n"
        )
        fixations = (
            "parcode,trial,fix_item,fix_time\n"
            "0,0,3,100\n0,0,0,50\n0,0,1,200\n0,0,0,50\n0,0,2,200\n"
            "0,1,0,50\n0,1,2,200\n0,1,0,50\n0,1,2,200\n"
        )
        rows, audit = convert_data(trials.encode(), fixations.encode())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["response"], 1)  # -1 is left in original file
        self.assertEqual(rows[0]["v0"], 1.0)       # right 15 degrees
        self.assertEqual(rows[0]["v1"], 10.0)      # left 0 degrees
        self.assertEqual(rows[0]["arr_ml3_left"], "[1, 0]")
        self.assertEqual(rows[0]["arr_ml3_time"], "[200, 200]")
        self.assertEqual(rows[0]["rt_raw_ms"], 600)
        self.assertEqual(rows[0]["rt_active_ms"], 400)
        self.assertEqual(rows[1]["response"], 0)
        self.assertEqual(rows[1]["arr_ml3_left"], "[0]")
        self.assertEqual(rows[1]["arr_ml3_time"], "[400]")
        self.assertTrue(audit["not_full_button_rt"])

    def test_input_mismatch_refused(self):
        trials = "parcode,trial,rt,choice,item_left,item_right,valid\n0,0,700,-1,0,5,\n"
        fx = "parcode,trial,fix_item,fix_time\n0,0,1,200\n"
        with self.assertRaises(ValueError):
            convert_data(trials.encode(), fx.encode())


if __name__ == "__main__":
    unittest.main()
