"""Pure stdlib contract checks; no model training or empirical data required."""
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pipeline as p

class TestParameterPipeline(unittest.TestCase):
    def test_eight_models(self):
        self.assertEqual(len(p.MODEL_NAMES),8)
        self.assertIn("aRACE_g",p.MODEL_NAMES)
        self.assertIn("aDDM_t",p.MODEL_NAMES)
        self.assertEqual(len(set(p.MODEL_NAMES)),8)

    def test_model_output_directories_isolated(self):
        paths=[p.destination(model,"formal") for model in p.MODEL_NAMES]
        self.assertEqual(len(set(paths)),8)
        for path in paths:
            self.assertTrue(str(path).startswith(str(p.OUTPUT)))

    def test_training_recovery_decoding_defaults(self):
        s1=p.parser("s1").parse_args(["--model","aRACE_g"])
        self.assertEqual(s1.batches*s1.batch_size,3200)
        self.assertEqual(s1.validation,2000)
        self.assertEqual(s1.epochs,5000)
        s2=p.parser("s2").parse_args(["--model","aDDM_t"])
        self.assertEqual(s2.n_test,2000)
        self.assertEqual(s2.draws,1000)
        s3=p.parser("s3").parse_args(["--model","aDDM_t"])
        self.assertEqual(s3.draws,2000)
        self.assertEqual(s3.ppc_draws,0)
        self.assertEqual(set(p.STAGE_SCRIPTS),{"s1","s2","s3"})

    def test_reject_bad_counts(self):
        a=p.parser("s1").parse_args(["--model","aDDM_t"])
        a.validation=0
        with self.assertRaises(ValueError):
            p.validate_args("s1",a)
        b=p.parser("s3").parse_args(["--model","aRACE_g"])
        b.ppc_draws=-1
        with self.assertRaises(ValueError):
            p.validate_args("s3",b)

if __name__=="__main__":
    unittest.main()
