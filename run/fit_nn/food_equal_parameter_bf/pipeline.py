"""Isolated eight-model BayesFlow parameter-inference driver.

Uses the ORIGINAL upstream S1/S2/S3 implementations (no reimplementation
of the data feature builder, model simulator, posterior or diagnostics).
"""
import argparse
import csv
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND","torch")
os.environ.setdefault("MPLBACKEND","Agg")
# Preserve the upstream dataset-specific value scaling (never force Tavares G2).
os.environ.setdefault("ATT_BF_VALUE_NORMALIZATION","none")

HERE=Path(__file__).resolve().parent
UPSTREAM=HERE.parent/"food_equal_bf"
ROOT=HERE.parents[2]
DATASET="food_equal"
INPUT=ROOT/"data"/DATASET/"trial_eye.csv"
OUTPUT=ROOT/"outputs"/(DATASET+"_parameter_bf")/"v1"
MODEL_NAMES=("aDDM_1","aDDM_2","aDDM_t","aDDM_g",
             "aRACE_1","aRACE_2","aRACE_t","aRACE_g")
STAGE_SCRIPTS={
    "s1":"s1_train_parameter_posterior.py",
    "s2":"s2_parameter_recovery.py",
    "s3":"s3_decode_real_data.py",
}

def check_static():
    """Check code/source paths before consuming any GPUs."""
    for stage,name in STAGE_SCRIPTS.items():
        if not (UPSTREAM/name).is_file():
            raise FileNotFoundError(f"Missing {stage} upstream implementation: {UPSTREAM/name}")
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing input CSV: {INPUT}; prepare data with upstream dataset script")
    with INPUT.open(newline="",encoding="utf8") as f:
        reader=csv.DictReader(f)
        if "subj" not in (reader.fieldnames or []):
            raise ValueError("Input lacks subj column")
        n=0; subjects=set()
        for row in reader:
            n+=1; subjects.add(str(row["subj"]))
    if n==0 or not subjects:
        raise ValueError("Empty prepared dataset")
    return {"dataset":DATASET,"input":str(INPUT),
            "input_sha256":hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            "prepared_trials":n,"subjects":len(subjects),
            "normalization":os.environ["ATT_BF_VALUE_NORMALIZATION"],
            "upstream":str(UPSTREAM)}

def parser(stage):
    p=argparse.ArgumentParser(description=f"{DATASET}: isolated BayesFlow parameter {stage}")
    p.add_argument("--model",choices=MODEL_NAMES,required=True)
    p.add_argument("--mode",choices=("formal","test"),default="formal")
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--dry-run",action="store_true")
    if stage=="s1":
        p.add_argument("--epochs",type=int,default=5000)
        p.add_argument("--batches",type=int,default=50)
        p.add_argument("--batch-size",type=int,default=64)
        p.add_argument("--validation",type=int,default=2000)
    elif stage=="s2":
        p.add_argument("--n-test",type=int,default=2000)
        p.add_argument("--draws",type=int,default=1000)
        p.add_argument("--batch-size",type=int,default=25)
    elif stage=="s3":
        p.add_argument("--draws",type=int,default=2000)
        p.add_argument("--batch-size",type=int,default=25)
        p.add_argument("--ppc-draws",type=int,default=0,
                       help="0: decode only; e.g. 100 to run expensive PPC")
    return p

def destination(model,mode):
    if model not in MODEL_NAMES:
        raise ValueError("Unknown model")
    return OUTPUT/model/mode

def validate_args(stage,a):
    if a.seed<0:
        raise ValueError("Seed must be nonnegative")
    required={
        "s1":("epochs","batches","batch_size","validation"),
        "s2":("n_test","draws","batch_size"),
        "s3":("draws","batch_size"),
    }[stage]
    if any(getattr(a,k)<1 for k in required):
        raise ValueError(f"{stage} requires strictly positive sizes")
    if stage=="s3" and a.ppc_draws<0:
        raise ValueError("PPC draws must be nonnegative")

def _posterior_spec(model):
    # Import ONLY the dataset-specific original model metadata. In food_equal
    # this contains the explicitly widened attention priors. No overrides.
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0,str(UPSTREAM))
    from _shared.data import model_infos, FEATURE_NAMES, SCALE
    if set(model_infos)!=set(MODEL_NAMES):
        raise RuntimeError("Upstream eight models have changed")
    info=model_infos[model]
    params=list(info["free_pnames"])
    bounds=[list(map(float,ab)) for ab in info["free_pranges"]]
    return {"parameters":params,"bounds":bounds,"feature_names":list(FEATURE_NAMES),
            "feature_scale":list(map(float,SCALE))}

def _check_checkpoint(model,mode,expected):
    folder=destination(model,mode)
    metadata=folder/"metadata.json"
    if not metadata.is_file() or not (folder/"best.keras").is_file():
        raise FileNotFoundError(f"No completed S1 checkpoint for {model}: {folder}")
    value=json.loads(metadata.read_text(encoding="utf8"))
    if (value.get("model")!=model or value.get("input_sha256")!=expected["input_sha256"]
        or value.get("parameter_names")!=expected["spec"]["parameters"]
        or value.get("features")!=expected["spec"]["feature_names"]):
        raise ValueError("S1 checkpoint differs from current model, features or input")
    if DATASET=="food_equal":
        if "prior_spec" not in value:
            raise ValueError("Missing food_equal BF prior_spec")
        prior=value["prior_spec"]["free_priors"][model]
        expected_priors=dict(zip(expected["spec"]["parameters"],expected["spec"]["bounds"]))
        if prior!=expected_priors:
            raise ValueError("S1 checkpoint priors have changed")
    elif (folder/"run_manifest.json").is_file():
        manifest=json.loads((folder/"run_manifest.json").read_text(encoding="utf8"))
        if manifest["spec"]!=expected["spec"]:
            raise ValueError("S1 frozen parameter bounds or feature scales changed")
    return value

def run(stage,a):
    validate_args(stage,a)
    expected=check_static()
    # Use the historical simulator/parameter metadata for this dataset.
    spec=_posterior_spec(a.model)
    expected["model"]=a.model
    expected["spec"]=spec
    expected["mode"]=a.mode
    folder=destination(a.model,a.mode)
    print("[PLAN]",json.dumps({**expected,"output":str(folder),
          "stage":stage},indent=2),flush=True)
    if a.dry_run:
        return
    if stage=="s1":
        if any((folder/name).exists() for name in
               ("best.keras","metadata.json","history.csv","run_manifest.json")):
            raise FileExistsError(f"Refusing to overwrite isolated S1 experiment: {folder}")
        folder.mkdir(parents=True,exist_ok=True)
        (folder/"run_manifest.json").write_text(
            json.dumps(expected,indent=2)+"\n",encoding="utf8")
    else:
        _check_checkpoint(a.model,a.mode,expected)
        out=folder/("diagnostics" if stage=="s2" else "real_data")
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing {stage}: {out}")

    argv=[STAGE_SCRIPTS[stage],"--model",a.model,"--mode",a.mode,
          "--input",str(INPUT),"--output",str(OUTPUT),
          "--seed",str(a.seed)]
    if stage=="s1":
        argv += ["--epochs",str(a.epochs),"--batches",str(a.batches),
                 "--batch-size",str(a.batch_size),"--validation",str(a.validation)]
    elif stage=="s2":
        argv += ["--n-test",str(a.n_test),"--draws",str(a.draws),
                 "--batch-size",str(a.batch_size)]
    else:
        argv += ["--draws",str(a.draws),"--batch-size",str(a.batch_size),
                 "--ppc-draws",str(a.ppc_draws)]
    before=sys.argv
    try:
        sys.argv=argv
        runpy.run_path(str(UPSTREAM/STAGE_SCRIPTS[stage]),run_name="__main__")
    finally:
        sys.argv=before
    if stage=="s1":
        _check_checkpoint(a.model,a.mode,expected)
        with (folder/"history.csv").open(newline="",encoding="utf8") as f:
            history=list(csv.DictReader(f))
        losses=[float(row["val_loss"]) for row in history]
        import math
        if not losses or not all(math.isfinite(x) for x in losses):
            raise RuntimeError("No finite validation loss / failed convergence audit")
        best_index=min(range(len(losses)),key=lambda i:losses[i])
        audit={"epochs_completed":len(losses),
               "best_epoch":best_index+1,
               "best_val_loss":losses[best_index],
               "final_val_loss":losses[-1],
               "stopped_before_max_epochs":len(losses)<a.epochs,
               "note":"Early stopping is not proof of global optimality"}
        (folder/"training_audit.json").write_text(
            json.dumps(audit,indent=2)+"\n",encoding="utf8")
        print("[TRAINING AUDIT]",audit,flush=True)
    print("[DONE]",stage,DATASET,a.model,folder,flush=True)
