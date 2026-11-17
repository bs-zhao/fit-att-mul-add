"""S3 gate: validate G2 prior predictive simulations before expensive S4.

Each model generates a few complete pseudo-subjects using the exact intended
clock, value scale, QC, 1-ms simulator, prior and 17-feature preprocessing.
This stage does NOT train a network, save weights or modify existing S4 data.
"""
import os
os.environ.setdefault("KERAS_BACKEND","torch")
os.environ["ATT_BF_VALUE_NORMALIZATION"]="dataset_max"
import argparse
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from _shared import common
from _shared.data import SCALE
from experiments import normalized_value_prior_sweep as sweep


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",choices=sorted(sweep.shared_data.VARIANTS),default="full_thomas_qc")
    p.add_argument("--n-per-model",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261117)
    args=p.parse_args()
    if args.n_per_model<1:
        p.error("--n-per-model must be >=1")
    sweep.configure_priors(1,50,.5,5,0,1,0,2)
    sweep.apply_shared_priors()
    common.environment(args.seed)
    path=common.ROOT / "data/tavares2017_v2" / args.profile / "trial_eye.csv"
    ex=common.Experiment(path)
    report={
        "profile":args.profile,"input_sha256":ex.sha256,
        "n_per_model":args.n_per_model,
        "model_prior_spec":sweep.prior_spec(common.MODEL_NAMES),
        "scale":SCALE.tolist(),
        "models":{},
    }
    for j,model in enumerate(common.MODEL_NAMES):
        gen,stats=ex.simulator(common.bf,model,args.seed+10000*j,True)
        batch=gen.sample(args.n_per_model)
        trials=np.asarray(batch["trials"])
        mask=np.asarray(batch["summary_mask"],dtype=bool)
        if trials.shape != (args.n_per_model,ex.max_trials,17):
            raise RuntimeError(f"{model} shape mismatch {trials.shape}")
        if not np.isfinite(trials).all() or not np.any(mask):
            raise RuntimeError(f"{model} nonfinite or empty")
        rt=trials[:,:,4][mask] * float(SCALE[4])
        per_subject=mask.sum(1)
        report["models"][model]={
            "sim_stats":stats,
            "included_trials":int(mask.sum()),
            "mean_trials_per_subject":float(np.mean(per_subject)),
            "mean_rt_seconds":float(np.mean(rt)),
            "median_rt_seconds":float(np.median(rt)),
            "p95_rt_seconds":float(np.quantile(rt,.95)),
            "max_rt_seconds":float(np.max(rt)),
            "nonhit_rate":stats["nonhits"]/max(stats["trials"],1),
            "retained_rate":int(mask.sum())/max(stats["trials"],1),
        }
        print("[PRIOR CHECK]",model,report["models"][model],flush=True)
    dest=common.ROOT/"outputs/tavares2017_bf_v2"/args.profile/"preflight"
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"prior_predictive.json").write_text(json.dumps(report,indent=2)+"\n")
    print("[READY]",dest/"prior_predictive.json",flush=True)


if __name__=="__main__":
    main()
