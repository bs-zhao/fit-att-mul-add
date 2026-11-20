"""Final1 no-GPU contract tests, across all three datasets."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import final1_launcher as L
import unified_final1 as F

class Final1Contract(unittest.TestCase):
    def test_priors(self):
        self.assertEqual(F.PRIORS,{
            "d":(.1,5.),"a":(.5,10.),
            "theta":(0.,1.),"gamma":(0.,10.),
        })
        self.assertEqual(len(F.MODELS),8)
        self.assertEqual(len(F.DATASETS),3)
        self.assertEqual(len(F.config_hash()),64)

    def test_six_stages_three_datasets(self):
        for d in F.DATASETS:
            root=ROOT/f"{d}_final1"
            self.assertTrue((root/"run.py").is_file())
            self.assertTrue((root/"submit.sh").is_file())
            for s in L.STAGES:
                path=root/f"{s}.py"
                self.assertTrue(path.is_file())
                source=path.read_text(encoding="utf8")
                compile(source,str(path),"exec")
                self.assertIn(f'["{s}","--dataset","{d}"',source)

    def test_outputs_isolated(self):
        for dataset in F.DATASETS:
            for stage,sel in (("s1",["--model","aDDM_t"]),
                              ("s2",["--model","aRACE_g"]),
                              ("s4",["--family","ddm"]),
                              ("s5",["--family","race"]),
                              ("s6",["--family","all"])):
                a=L.parse([stage,"--dataset",dataset,*sel])
                out=L.folder(a)
                self.assertIn(f"outputs/{dataset}_final1",str(out))
                self.assertNotIn("unified_g2_v1",str(out))
                cli=L.upstream_cli(a)
                self.assertIn(str(ROOT.parents[1]/"outputs"/f"{dataset}_final1"),cli)
        a=L.parse(["s4","--dataset","tavares2017","--family","all"])
        self.assertEqual(L.upstream_cli(a)[-8:],
                         ["--epochs","5000","--batches","128",
                          "--batch-size","8","--validation","512"])

    def test_upstream_sources_exist(self):
        for dataset in F.DATASETS:
            for stage,script in L.STAGES.items():
                with self.subTest(dataset=dataset,stage=stage):
                    self.assertTrue((ROOT/f"{dataset}_bf"/script).is_file())

    def test_reject_wrong_stage(self):
        with self.assertRaises(SystemExit):
            L.parse(["s4","--dataset","food_equal","--model","aDDM_t"])
        with self.assertRaises(SystemExit):
            L.parse(["s2","--dataset","food_equal","--family","all"])
        with self.assertRaises(SystemExit):
            L.parse(["s3","--dataset","food_equal","--model","aRACE_g",
                     "--ppc-draws","-1"])

    def test_same_model_spec_for_every_dataset(self):
        # Import shared algebraic spec without BayesFlow/Cython.
        import runpy
        from copy import deepcopy
        g2=runpy.run_path(str(ROOT/"unified_g2.py"))
        base=g2["model_specifications"]()
        for name in F.MODELS:
            info=base[name]
            bounds=[F.PRIORS[x] for x in info["free_pnames"]]
            info["free_pranges"]=bounds
            info["free_pranges_ml"]=[F.PRIORS[x] for x in info["free_pnames_ml"]]
        self.assertTrue(F.validate_models(base))
        self.assertTrue(F.ensure_equal_cross_dataset_specs([deepcopy(base) for _ in F.DATASETS]))

if __name__=="__main__":
    unittest.main()
