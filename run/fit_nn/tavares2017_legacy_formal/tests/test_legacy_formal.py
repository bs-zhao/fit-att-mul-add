"""No-GPU contract tests for isolated legacy formal Tavares G2 pipeline."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core


class TestLegacyFormal(unittest.TestCase):
    def test_three_family_paths_and_unchanged_eight_models(self):
        fams=("ddm","race","all")
        self.assertEqual(len({core.directory(f) for f in fams}),3)
        self.assertEqual(len(core.FAMILIES["all"]),8)
        self.assertEqual(core.FAMILIES["ddm"],("aDDM_1","aDDM_2","aDDM_t","aDDM_g"))
        self.assertEqual(core.FAMILIES["race"],("aRACE_1","aRACE_2","aRACE_t","aRACE_g"))
        self.assertTrue(all(str(core.directory(f)).startswith(str(core.OUTPUT)) for f in fams))

    def test_old_input_hash_and_expected_trial_count(self):
        data=core.validate_input()
        self.assertEqual(data["sha256"],core.EXPECTED_SHA)
        self.assertEqual(data["prepared_trials"],31854)
        self.assertEqual(data["retained_after_existing_17feature_rule"],31853)

    def test_s4_defaults_and_stages_are_independent(self):
        a=core.parser("s4").parse_args(["--family","ddm"])
        self.assertEqual(a.batches*a.batch_size,1024)
        self.assertEqual(a.validation,512)
        self.assertEqual(a.early_patience,30)
        b=core.parser("s5").parse_args(["--family","race"])
        self.assertEqual(b.n_per_model,2000)
        c=core.parser("s6").parse_args(["--family","all"])
        self.assertEqual(c.batch_size,4)
        with self.assertRaises(ValueError):
            a.early_patience=4
            core.args_check(a,"s4")

    def test_finite_validation_and_best_epoch_audit(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            (folder/"best.keras").write_bytes(b"test-checkpoint")
            with (folder/"history.csv").open("w",newline="") as f:
                wr=csv.writer(f)
                wr.writerow(("epoch","loss","val_loss"))
                wr.writerow((0,.5,.35))
                wr.writerow((1,.4,.2))
                wr.writerow((2,.35,.25))
            rec=core.audit_training(folder,5000)
            self.assertEqual(rec["epochs_completed"],3)
            self.assertEqual(rec["best_epoch"],2)
            self.assertTrue(rec["early_stop_before_maximum"])
            self.assertEqual(json.loads((folder/"formal_training_audit.json").read_text())["best_epoch"],2)

    def test_nonfinite_validation_fails(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            (folder/"best.keras").write_bytes(b"test")
            (folder/"history.csv").write_text("epoch,val_loss\n0,nan\n")
            with self.assertRaisesRegex(RuntimeError,"Nonfinite"):
                core.audit_training(folder,5000)


if __name__=="__main__":
    unittest.main()
