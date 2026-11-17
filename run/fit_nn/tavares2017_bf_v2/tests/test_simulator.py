"""Cython parity of all 8 binary models; deliberate zero drift under neutral."""
import ctypes
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT / "packages/c_bifood"))
from _shared.data import Experiment, feature, VARIANTS, model_infos
from c_bifood import sim_trial_aDDM as old_ddm, sim_trial_aRACE as old_race
from c_tavares_fulltimeline import sim_trial_aDDM as new_ddm, sim_trial_aRACE as new_race


def seed(k):
    ctypes.CDLL(None).srand(k)


class Simulators(unittest.TestCase):
    def params(self, model):
        info=model_infos[model]
        p = dict(zip(info["fixed_pnames"],info["fixed_pvalues"]))
        p.update(dict(d=5., a=1.5))
        if "theta" in info["free_pnames"]: p["theta"] = .5
        if "gamma" in info["free_pnames"]: p["gamma"] = 1.
        return p

    def test_all_eight_binary_parity(self):
        v=np.array([.4,.8],np.float64)
        f=np.array([1,0,1,0,1],np.int64)
        t=np.array([170,220,140,500,100],np.int64)
        for m in model_infos:
            params=self.params(m)
            old=old_ddm if m.startswith("aDDM") else old_race
            new=new_ddm if m.startswith("aDDM") else new_race
            seed(12345)
            a=old(params,v,f,t,extend_last=10000,dt=.001,max_rt=10.,rcd=1)
            seed(12345)
            b=new(params,v,f,t,extend_last=10000,dt=.001,max_rt=10.,rcd=1)
            self.assertEqual(a[0],b[0],m)
            if m.startswith("aDDM"):
                for i in range(4): np.testing.assert_array_equal(a[1][i],b[1][i],err_msg=m)
            else:
                for i in range(2):
                    np.testing.assert_array_equal(a[1][0][i],b[1][0][i],err_msg=m)
                for i in range(1,4):
                    np.testing.assert_array_equal(a[1][i],b[1][i],err_msg=m)

    def test_neutral_zero_drift_with_noise_disabled(self):
        f=np.array([-1,1],np.int64)
        t=np.array([400,1000],np.int64)
        v=np.array([.2,.9],np.float64)
        for m in model_infos:
            p=self.params(m)
            p["s"]=0.
            p["a"]=.5
            p["d"]=5.
            seed(11)
            sim=new_ddm if m.startswith("aDDM") else new_race
            result=sim(p,v,f,t,extend_last=2000,dt=.001,max_rt=3.,rcd=1)
            self.assertGreater(result[0][1],.4,m)
            if m.startswith("aDDM"):
                np.testing.assert_array_equal(result[1][0][:400],np.zeros(400),err_msg=m)
            else:
                np.testing.assert_array_equal(result[1][0][0][:400],np.zeros(400),err_msg=m)
                np.testing.assert_array_equal(result[1][0][1][:400],np.zeros(400),err_msg=m)

    def test_real_encoding_and_masks(self):
        for name in ("active_thomas_all","full_thomas_all",
                     "active_thomas_qc","full_thomas_qc",
                     "active_relative_qc","full_relative_qc"):
            p=ROOT / "data/tavares2017_v2" / name / "trial_eye.csv"
            ex=Experiment(p)
            self.assertEqual(len(ex.groups),25)
            self.assertEqual(ex.value_max,10. if "thomas" in name else 1.)
            ids, raw, audit=ex.real()
            self.assertEqual(len(ids),25)
            self.assertEqual(raw["trials"].shape[1:],(ex.max_trials,17))
            self.assertTrue(raw["summary_mask"].any())
            self.assertTrue(np.isfinite(raw["trials"]).all())
            for val in ex.df["arr_pos"]:
                self.assertTrue(set(val).issubset({-1,0,1}))
                if name.startswith("active_"): self.assertNotIn(-1,val)


if __name__=="__main__":
    unittest.main()
