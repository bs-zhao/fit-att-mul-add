"""Offline checks for the single final-data S4/S5/S6 entry point.

These are pure stdlib and run under GitHub Actions without a GPU/BayesFlow.
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parents[1]
BF_DIR=HERE.parent
sys.path.insert(0,str(BF_DIR))
sys.path.insert(0,str(HERE))
from final_training.plan import INPUT, OUTPUT, FAMILIES, classifier_dir, verify_final_input
from final_training.run import parse_args, plan, _patch_isolated_output
from report import data_from_folder


class TestFinalStudy(unittest.TestCase):
    def test_three_classifier_paths_are_unique_and_isolated(self):
        paths=[classifier_dir(f) for f in ("ddm","race","all")]
        self.assertEqual(len(set(paths)),3)
        self.assertEqual(paths[2],OUTPUT/"classifier/formal")
        self.assertTrue(all(str(p).startswith(str(OUTPUT)) for p in paths))
        self.assertEqual(len(FAMILIES["all"]),8)
        self.assertEqual(len(FAMILIES["race"]),4)

    def test_checked_final_data_sha_qc_and_all_trials(self):
        info=verify_final_input()
        self.assertEqual(info["n_trials"],30162)
        self.assertEqual(info["n_subjects"],25)
        self.assertEqual(info["input"],str(INPUT.resolve()))
        self.assertEqual(len(info["input_sha256"]),64)

    def test_guards_reject_changed_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"trial_eye.csv"
            m=Path(tmp)/"prep_manifest.json"
            p.write_bytes(INPUT.read_bytes()+b"\n")
            m.write_text((INPUT.parent/"prep_manifest.json").read_text())
            with self.assertRaisesRegex(ValueError,"SHA256 differs"):
                verify_final_input(p,m)

    def test_default_training_config(self):
        args=parse_args(["--stage","s4","--family","race"])
        info=plan(args)
        self.assertEqual(info["training_samples_per_epoch"],1024)
        self.assertEqual(info["validation_subjects"],512)
        self.assertEqual(info["early_stopping_patience"],20)
        self.assertEqual(info["reduce_lr_patience"],6)
        self.assertIn("aRACE_t",info["models"])
        self.assertTrue(info["checkpoint_dir"].endswith(
            "/classifier/aRACE_1__aRACE_2__aRACE_t__aRACE_g/formal"))

    def test_test_mode_small_and_no_confounding_modes(self):
        args=parse_args(["--stage","s4","--family","ddm","--mode","test"])
        self.assertEqual(args.batches,2)
        self.assertEqual(args.batch_size,2)
        self.assertEqual(args.validation,8)
        with self.assertRaises(SystemExit):
            parse_args(["--stage","s6","--family","all","--batch-size","8"])
        with self.assertRaises(SystemExit):
            parse_args(["--stage","s5","--family","all","--validation","512"])

    def test_fixed_priors_and_output_redirection(self):
        calls=[]
        mock=SimpleNamespace(OUTPUT=Path("/tmp/old"),
                             configure_priors=lambda *args:calls.append(args))
        _patch_isolated_output(mock)
        mock.configure_priors(1,50,.5,5,0,1,0,2)
        self.assertEqual(mock.OUTPUT,OUTPUT)
        self.assertEqual(calls,[(1.,50.,.5,5.,0.,1.,0.,2.)])
        with self.assertRaisesRegex(ValueError,"unchanged G2"):
            mock.configure_priors(1,50,.5,5,0,1,0,3)
        self.assertEqual(len(calls),1)

    def test_read_only_recovery_summary_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)
            (f/"diagnostics").mkdir()
            (f/"real_data").mkdir()
            (f/"metadata.json").write_text(json.dumps({
                "model_names":FAMILIES["ddm"],"input_sha256":"fixture",
                "config":{"batch_size":8},"early_stopping_patience":20,
            }))
            (f/"diagnostics/metrics.json").write_text(json.dumps({
                "accuracy":.925,"family_accuracy":1.,"ece":.02,
            }))
            with (f/"diagnostics/predictions.csv").open("w",newline="") as handle:
                w=csv.writer(handle)
                w.writerow(["true_model","pred_model"])
                for _ in range(80):
                    w.writerow(["aDDM_1","aDDM_1"])
            (f/"diagnostics/per_model_metrics.csv").write_text(
                "model,n,recall\naDDM_1,20,0.9\n")
            (f/"real_data/model_probabilities.csv").write_text(
                "subj,best_model\n1,aDDM_1\n2,aDDM_t\n")
            info=data_from_folder(f,"ddm")
            self.assertEqual(info["accuracy"],.925)
            self.assertEqual(info["n_recovery_per_model"],20)
            self.assertEqual(info["s6_winners"]["aDDM_1"],1)
            self.assertEqual(info["early_stopping_patience"],20)


if __name__=="__main__":
    unittest.main()
