"""Unit tests for one non-configurable Tavares preprocessing standard."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from prepare_data import convert_data
from final_preprocessing.prepare_final import (
    EXPECTED, _csv_bytes, assemble,
    feature_limit_reason, independent_eye_encoding,
    validate_thomas_reference,
)


def fixture():
    exp=(
        "parcode,trial,rt,choice,item_left,item_right,valid\n"
        "0,0,1000,-1,0,15,\n"
        "0,1,1000,1,5,10,\n"
        "1,0,30000,-1,-5,0,\n"
        "1,1,1000,1,10,5,\n"
        "1,2,1000,-1,0,5,\n"
    )
    eyes=(
        "parcode,trial,fix_item,fix_time\n"
        "0,0,3,100\n0,0,0,100\n0,0,1,300\n0,0,0,100\n0,0,2,400\n"
        "0,1,3,600\n0,1,1,100\n0,1,0,200\n0,1,2,100\n"
        "1,0,1,30000\n"
        "1,1,3,100\n1,1,2,200\n1,1,0,50\n1,1,2,200\n1,1,1,450\n"
        "1,2,3,200\n1,2,1,250\n1,2,0,300\n1,2,2,250\n"
    )
    exp_bytes,eyes_bytes=exp.encode(),eyes.encode()
    expected=dict(EXPECTED,original_trials=5,original_fixations=19,
                  subjects=2,offitem_qc_exclusions=1,
                  feature_exclusions=1,final_trials=3)
    return exp_bytes, eyes_bytes, expected


class TestSingleFinalPreprocess(unittest.TestCase):
    def test_gaze_codes_and_merge(self):
        x,stats=independent_eye_encoding([
            (3,100),(0,200),(1,50),(0,100),(1,75),(2,300)])
        self.assertEqual(x,[[1,125],[0,300]])
        self.assertEqual(stats[0]+stats[3],400)

    def test_exact_feature_cutoff(self):
        self.assertEqual(feature_limit_reason([],[]),"no_item_gaze")
        self.assertEqual(feature_limit_reason([1]*128,[1]*128),"feature_fixation_count")
        self.assertEqual(feature_limit_reason([1],[28285]),"feature_rt_limit")
        self.assertEqual(feature_limit_reason([1],[28284]),"")
        self.assertEqual(feature_limit_reason([1,0],[100,100]),"")

    def test_full_pipeline_fixed_qc_and_choice(self):
        exp,eyes,expected=fixture()
        with patch.dict(EXPECTED,expected):
            kept,audit,subjects,summary,all_prepared=assemble(exp,eyes)
        self.assertEqual(len(kept),3)
        self.assertEqual(summary["excluded_by_reason"]["offitem_qc_gt50pct"],1)
        self.assertEqual(summary["excluded_by_reason"]["feature_rt_limit"],1)
        keys={(r["subj"],r["trial"]) for r in kept}
        self.assertEqual(keys,{(0,0),(1,1),(1,2)})
        a={(r["subj"],r["trial"]):r for r in audit}
        self.assertEqual(a[(0,1)]["exclude_reason"],"offitem_qc_gt50pct")
        self.assertEqual(a[(1,0)]["exclude_reason"],"feature_rt_limit")
        self.assertEqual(a[(1,2)]["exclude_reason"],"")
        self.assertEqual(a[(1,2)]["off_item_fraction"],.5)
        self.assertEqual(kept[0]["v0"],1.)
        self.assertEqual(kept[0]["v1"],10.)
        self.assertEqual(kept[0]["response"],1)
        self.assertEqual(kept[0]["rt_active_ms"],700)
        self.assertEqual(kept[0]["rt_raw_ms"],1000)
        self.assertEqual(kept[1]["arr_ml3_left"],"[0, 1]")
        self.assertEqual(kept[1]["arr_ml3_time"],"[400, 450]")

    def test_thomas_reference_is_independent_check(self):
        exp,eyes,expected=fixture()
        all_prepared,_=convert_data(exp,eyes)
        rows=[]
        for r in all_prepared:
            p=__import__("ast").literal_eval(r["arr_ml3_left"])
            d=__import__("ast").literal_eval(r["arr_ml3_time"])
            left=sum(ms for pos,ms in zip(p,d) if pos==1)
            right=sum(ms for pos,ms in zip(p,d) if pos==0)
            rows.append({
                "subject":r["subj"],"trial":r["trial"],"rt":r["rt_raw_ms"],
                "choice":1-r["response"],
                "item_value_0":r["v1"],"item_value_1":r["v0"],
                "gaze_0":left/(left+right),"gaze_1":right/(left+right),
            })
        reference=_csv_bytes(rows,list(rows[0]))
        self.assertEqual(validate_thomas_reference(all_prepared,reference)["mismatch_count"],0)
        bad=copy.deepcopy(rows)
        bad[0]["choice"]=1-bad[0]["choice"]
        with self.assertRaisesRegex(AssertionError,"choice mismatch"):
            validate_thomas_reference(all_prepared,_csv_bytes(bad,list(bad[0])))


if __name__=="__main__":
    unittest.main()
