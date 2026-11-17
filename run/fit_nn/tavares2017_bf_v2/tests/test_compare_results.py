"""Pure CPU fixture tests for S5/S6 comparisons and posterior conditioning."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compare_results import MODELS, folder, load, paired, global_summary, comparisons


class TestReports(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        self.samples={}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self,fam,probabilities,ids=("0","1")):
        directory=folder(self.base,fam)
        (directory/"real_data").mkdir(parents=True)
        (directory/"diagnostics").mkdir(parents=True)
        models=MODELS[fam]
        with (directory/"real_data/model_probabilities.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["subj"]+models)
            for sid,prob in zip(ids,probabilities):
                writer.writerow([sid]+list(prob))
        (directory/"metadata.json").write_text(json.dumps({
            "model_names":models,"config":{"batch_size":8},
            "input_sha256":"test_sha"}))
        (directory/"diagnostics/metrics.json").write_text(json.dumps({
            "accuracy":.9,"family_accuracy":1., "ece":.02,
        }))

    def test_joint_conditional_consistency(self):
        ddm=np.array([[.1,.2,.3,.4],[.4,.3,.2,.1]])
        race=np.array([[.4,.3,.2,.1],[.1,.2,.3,.4]])
        masses=np.array([.2,.75])
        joint=np.concatenate([ddm*masses[:,None], race*(1-masses[:,None])],axis=1)
        for fam,arr in (("ddm",ddm),("race",race),("all",joint)):
            self.build(fam,arr)
        loaded={fam:load(self.base,fam) for fam in MODELS}
        rows=paired("fixture",loaded)
        self.assertEqual(len(rows),4)
        self.assertLess(max(r["TV"] for r in rows),1e-10)
        self.assertTrue(all(r["winner_agreement"] for r in rows))
        summary=global_summary("fixture","all",loaded["all"])
        self.assertAlmostEqual(summary["mean_p_DDM"],np.mean(masses))
        self.assertEqual(summary["n_subjects"],2)
        self.assertEqual(comparisons({"p1":loaded,"p2":loaded})[0]["mean_subject_TV"],0.)

    def test_missing_model_name_rejected(self):
        self.build("ddm",[[.5,.5,.0,.0]])
        path=folder(self.base,"ddm")/"metadata.json"
        path.write_text(json.dumps({"model_names":["wrong"]}))
        with self.assertRaises(ValueError):
            load(self.base,"ddm")


if __name__=="__main__":
    unittest.main()
