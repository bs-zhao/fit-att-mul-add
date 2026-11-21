"""Final1 cross-architecture check: generate aRACE_g, classify by four DDM.

No RACE classifier checkpoint is required. All generation and inference uses
the SAME Final1 value scale and prior profile as food_equal_final1 S4.
"""
import os
os.environ.setdefault("KERAS_BACKEND","torch")
os.environ.setdefault("MPLBACKEND","Agg")
os.environ["ATT_BF_VALUE_NORMALIZATION"]="dataset_max"

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIT_NN=HERE.parent
UPSTREAM=FIT_NN/"food_equal_bf"
REPO=FIT_NN.parents[1]
sys.path.insert(0,str(FIT_NN))
sys.path.insert(0,str(UPSTREAM))
from unified_final1 import install, audit_experiment, PRIORS, PROFILE

DDM=["aDDM_1","aDDM_2","aDDM_t","aDDM_g"]
LABELS=["NoAtt","AttOnly","Mul","Add"]
GENERATOR="aRACE_g"

def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=REPO/"data/food_equal/trial_eye.csv")
    p.add_argument("--output",type=Path,default=REPO/"outputs/food_equal_final1")
    p.add_argument("--mode",choices=("test","formal"),default="formal")
    p.add_argument("--n",type=int,default=1000,help="Accepted simulated participants")
    p.add_argument("--batch-size",type=int,default=25)
    p.add_argument("--seed",type=int,default=20261102)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args(argv)
    if min(args.n,args.batch_size)<1 or args.seed<0:
        p.error("--n/--batch-size must be positive, seed nonnegative")
    return args

def main(argv=None):
    args=arguments(argv)
    # Patch the shared g2 instance BEFORE importing the BayesFlow workflow.
    source=importlib.import_module("_shared.data")
    install(source.unified_g2)
    ex=source.Experiment(args.input,unified=True)
    audit=audit_experiment(ex)
    folder=args.output/"classifier"/"__".join(DDM)/args.mode
    dest=args.output/"cross_architecture"/"generated_aRACE_g__tested_DDM4"/args.mode/f"seed_{args.seed}"
    print("[FINAL1 CROSS PREFLIGHT]",json.dumps({
        "generator":GENERATOR,"classifier":DDM,
        "priors":PRIORS,"dataset":audit,"checkpoint":str(folder),
        "destination":str(dest)},indent=2),flush=True)
    if args.dry_run:
        return
    from _shared.common import bf, keras, dump, load
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from experiments.race_add_cross_architecture import classify
    from _shared.diagnostics import heatmap

    if not (folder/"best.keras").is_file():
        raise FileNotFoundError(f"Train Final1 DDM4 S4 first: {folder}")
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf8"))
    source.unified_g2.validate_checkpoint(meta,ex,DDM,source.FEATURE_NAMES,folder)
    if meta.get("model_names")!=DDM:
        raise ValueError("Not the canonical four-DDM checkpoint")
    if not np.allclose(meta.get("model_prior"),[.25]*4):
        raise ValueError("DDM4 training prior is not uniform")

    from _shared.common import environment
    environment(args.seed)
    # Load exact Final1 checkpoint. Do NOT route through the old non-unified loader.
    estimator=keras.saving.load_model(folder/"best.keras")

    sim, counters=ex.simulator(bf,GENERATOR,args.seed+800000,model_only=False)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Cross experiment already exists: {dest}")
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for start in range(0,args.n,args.batch_size):
        raw=sim.sample(min(args.batch_size,args.n-start))
        conditions={k:raw[k] for k in ("trials","summary_mask","n_trials")}
        probs, log_bf=classify(estimator,conditions,DDM)
        for j in range(len(probs)):
            rec={
                "sim_id":start+j,
                "generator":GENERATOR,
                "true_d":float(np.asarray(raw["d"])[j].reshape(-1)[0]),
                "true_a":float(np.asarray(raw["a"])[j].reshape(-1)[0]),
                "true_gamma":float(np.asarray(raw["gamma"])[j].reshape(-1)[0]),
                "n_retained_trials":int(np.asarray(raw["summary_mask"][j]).sum()),
                "DDM_logBF_Mul_Add":float(log_bf[j]),
                "DDM_best_model":DDM[int(probs[j].argmax())],
            }
            rec.update({f"DDM_p_{name}":float(probs[j,i]) for i,name in enumerate(LABELS)})
            rows.append(rec)
        print(f"[CROSS] {min(start+args.batch_size,args.n)}/{args.n}",flush=True)
    frame=pd.DataFrame(rows)
    values=frame.DDM_logBF_Mul_Add.to_numpy()
    wins=frame.DDM_best_model.value_counts()
    summary={
        "experiment":"food_equal_final1_race_add_to_ddm4",
        "profile":PROFILE,
        "generator":GENERATOR,
        "generator_prior":{p:list(PRIORS[p]) for p in ("d","a","gamma")},
        "ddm_models":DDM,
        "trained_model_prior":[.25]*4,
        "n":len(frame),
        "seed":args.seed,
        "mean_logBF_Mul_Add":float(values.mean()),
        "median_logBF_Mul_Add":float(np.median(values)),
        "Mul_wins_over_Add":int((values>0).sum()),
        "Add_wins_over_Mul":int((values<0).sum()),
        "DDM_winner_counts":{name:int(wins.get(name,0)) for name in DDM},
        "DDM_mean_probabilities":{name:float(frame[f"DDM_p_{label}"].mean()) for name,label in zip(DDM,LABELS)},
        "input_sha256":ex.sha256,
        "source_value_max":ex.value_max,
        "checkpoint":str(folder/"best.keras"),
        "simulation_stats":counters,
    }
    frame.to_csv(dest/"per_simulation.csv",index=False)
    dump(dest/"summary.json",summary)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(values,bins=40)
    ax.axvline(0,color="gray",ls="--")
    ax.set(xlabel="DDM log BF (Mul / Add)",ylabel="Simulated participants")
    fig.tight_layout()
    fig.savefig(dest/"ddm_logBF_mul_add.png",dpi=160)
    plt.close(fig)
    print("[CROSS RESULT]",json.dumps(summary,indent=2),flush=True)
    print("[OUTPUT]",dest,flush=True)

if __name__=="__main__":
    main()
