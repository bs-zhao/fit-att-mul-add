"""Pure-stdlib checks for controlled 2x2 G2 S6 inference and summary."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BF_DIR=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BF_DIR))
from final_training import cross_inference as c
from final_training import cross_report as r


def make_cell_csv(path,models,count_mul):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=["subj",*models,"best_model"])
        writer.writeheader()
        for sid in range(25):
            model=models[2] if sid<count_mul else models[3]
            probs={m:.05 for m in models}
            probs[model]=.85
            writer.writerow({"subj":sid,**probs,"best_model":model})


class TestCrossPipeline(unittest.TestCase):
    def test_pinned_sources_match_rebuilt_original_and_final(self):
        # CI creates both input files before running these tests.
        self.assertEqual(c.sha256(c.OLD_DATA),c.DATA_SHA["old"])
        self.assertEqual(c.sha256(c.NEW_DATA),c.DATA_SHA["new"])
        self.assertEqual(c.TRIALS,{"old":31853,"new":30162})
        self.assertEqual(c.MAX_TRIALS,{"old":1342,"new":1339})

    def test_checkpoint_output_stays_separate_from_training(self):
        for f in ("ddm","race","all"):
            for ck,dat in (("old","new"),("new","old")):
                path=c.cell_dir(f,ck,dat)
                self.assertTrue(str(path).startswith(str(c.CROSS_OUTPUT)))
                self.assertNotEqual(path,c.checkpoint_folder(ck,f))
                self.assertEqual(len(c.FAMILIES[f]),8 if f=="all" else 4)

    def test_2x2_decomposition_mathematically_reconstructs_total(self):
        group={
            "ddm":{
                "old_old":{"mul_winners":12,"add_winners":13,"mean_p_mul":.39,"mean_p_add":.54},
                "old_new":{"mul_winners":10,"add_winners":15,"mean_p_mul":.31,"mean_p_add":.62},
                "new_old":{"mul_winners":6,"add_winners":19,"mean_p_mul":.24,"mean_p_add":.66},
                "new_new":{"mul_winners":4,"add_winners":21,"mean_p_mul":.16,"mean_p_add":.74},
            }
        }
        out=r.factorial_decomposition(group)
        mul=next(x for x in out if x["metric"]=="mul_winners")
        self.assertEqual(mul["data_effect_at_old_classifier_B_minus_A"],-2)
        self.assertEqual(mul["classifier_effect_on_old_data_C_minus_A"],-6)
        self.assertEqual(mul["interaction_D_minus_C_minus_B_plus_A"],0)
        self.assertEqual(mul["total_change_D_minus_A"],-8)

    def test_four_cells_data_loaded_and_ddm_race_contrast(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            def ckdir(checkpoint,family):
                return root/"checkpoints"/checkpoint/family
            def xdir(family,checkpoint,data):
                return root/"cross"/family/(checkpoint+"_"+data)
            wins={
                "ddm":{"old_old":12,"old_new":9,"new_old":6,"new_new":4},
                "race":{"old_old":9,"old_new":8,"new_old":6,"new_new":5},
            }
            for family in ("ddm","race"):
                names=c.FAMILIES[family]
                for ck,dat in r.FOUR_CELLS:
                    key=ck+"_"+dat
                    folder=ckdir(ck,family) if ck==dat else xdir(family,ck,dat)
                    path=(folder/"real_data/model_probabilities.csv"
                        if ck==dat else folder/"model_probabilities.csv")
                    make_cell_csv(path,names,wins[family][key])
                    if ck==dat:
                        metadata={"input_sha256":c.DATA_SHA[dat]}
                    else:
                        metadata={
                            "dataset_sha256":c.DATA_SHA[dat],
                            "checkpoint_training_data_sha256":c.DATA_SHA[ck],
                            "family":family,"checkpoint":ck,"dataset":dat,
                            "n_subjects":25,
                            "n_retained_trials":c.TRIALS[dat],
                        }
                    (folder/"metadata.json").write_text(json.dumps(metadata))
            with patch.object(r,"checkpoint_folder",side_effect=ckdir),\
                 patch.object(r,"cell_dir",side_effect=xdir):
                stats,subject,table,sources=r.summaries(["ddm","race"])
            self.assertEqual(len(stats),8)
            self.assertEqual(len(subject),200)
            self.assertEqual(table["ddm"]["old_old"]["mul_winners"],12)
            self.assertEqual(table["ddm"]["new_new"]["mul_winners"],4)
            comparison=r.ddm_race_comparison(table)
            self.assertEqual(len(comparison),4)
            self.assertEqual(comparison[0]["ddm_minus_race_mul_winners"],3)
            self.assertEqual(comparison[3]["ddm_minus_race_mul_winners"],-1)
            self.assertEqual(len(sources["ddm"]),4)

    def test_cross_metadata_mismatch_must_fail(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            make_cell_csv(folder/"model_probabilities.csv",c.FAMILIES["ddm"],12)
            (folder/"metadata.json").write_text(json.dumps({
                "family":"ddm","checkpoint":"old","dataset":"new",
                "dataset_sha256":"wrong",
                "checkpoint_training_data_sha256":c.DATA_SHA["old"],
                "n_subjects":25,"n_retained_trials":c.TRIALS["new"],
            }))
            with patch.object(r,"cell_dir",return_value=folder):
                with self.assertRaisesRegex(ValueError,"Incorrect crossed"):
                    r.verified_cell("ddm","old","new")


if __name__=="__main__":
    unittest.main()
