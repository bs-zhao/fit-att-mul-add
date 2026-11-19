"""Frozen original Tavares2017 G2 model pipeline; new independent formal run.

The behavior, eye clock, 17 BF features and eight Cython models are imported
from the legacy Tavares2017 BF implementation. Do not refactor them here.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND","torch")
os.environ.setdefault("MPLBACKEND","Agg")
os.environ["ATT_BF_VALUE_NORMALIZATION"]="dataset_max"

HERE=Path(__file__).resolve().parent
BF_DIR=HERE.parent/"tavares2017_bf"
ROOT=HERE.parents[2]
sys.path.insert(0,str(BF_DIR))

INPUT=ROOT/"data/tavares2017/trial_eye.csv"
OUTPUT=ROOT/"outputs/tavares2017_legacy_formal/v1"
EXPECTED_SHA="ff1479b10957aca922ec0792d87cb5bda0f6b19374d27c1eb6edc87c40bbf5be"
EXPECTED_COUNTS={"subjects":25,"prepared_trials":31854,"bf_retained_trials":31853,"max_trials":1342}
FAMILIES={
    "ddm":("aDDM_1","aDDM_2","aDDM_t","aDDM_g"),
    "race":("aRACE_1","aRACE_2","aRACE_t","aRACE_g"),
}
FAMILIES["all"]=FAMILIES["ddm"]+FAMILIES["race"]

def directory(family,mode="formal"):
    if family not in FAMILIES or mode not in ("test","formal"):
        raise ValueError("Unknown model family or mode")
    if family=="all":
        return OUTPUT/"classifier"/mode
    return OUTPUT/"classifier"/"__".join(FAMILIES[family])/mode

def validate_input():
    """Lightweight pre-Slurm audit without BayesFlow or importing heavy libs."""
    import csv
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing legacy input CSV: {INPUT}; run original prepare_data.py")
    digest=hashlib.sha256(INPUT.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA:
        raise ValueError(f"Legacy input changed: SHA256 {digest}, expected {EXPECTED_SHA}")
    with INPUT.open(newline="",encoding="utf8") as f:
        rdr=csv.DictReader(f)
        n=0; subjects=set()
        for row in rdr:
            n+=1; subjects.add(str(row["subj"]))
    if n!=31854 or len(subjects)!=25:
        raise ValueError(f"Wrong legacy input counts {n}/{len(subjects)}")
    return {"file":str(INPUT),"sha256":digest,"prepared_trials":n,"subjects":len(subjects),
            "retained_after_existing_17feature_rule":31853,
            "max_trials":1342,"normalization":"dataset_max",
            "clock":"option fixation active time; no >50% off-item exclusion"}

def parser(stage):
    p=argparse.ArgumentParser(description="Original Tavares G2 formal "+stage)
    p.add_argument("--family",choices=("ddm","race","all"),required=True)
    p.add_argument("--mode",choices=("formal","test"),default="formal")
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--dry-run",action="store_true")
    if stage=="s4":
        p.add_argument("--epochs",type=int,default=5000)
        p.add_argument("--batches",type=int,default=128)
        p.add_argument("--batch-size",type=int,default=8)
        p.add_argument("--validation",type=int,default=512)
        p.add_argument("--sim-workers",type=int,default=8)
        p.add_argument("--lr-patience",type=int,default=8)
        p.add_argument("--early-patience",type=int,default=30)
    elif stage=="s5":
        p.add_argument("--n-per-model",type=int,default=2000)
        p.add_argument("--batch-size",type=int,default=4)
    elif stage=="s6":
        p.add_argument("--batch-size",type=int,default=4)
    return p

def args_check(args,stage):
    if args.seed<0 or any(getattr(args,k,1)<1 for k in
           (("epochs","batches","batch_size","validation","sim_workers",
             "lr_patience","early_patience") if stage=="s4" else
            ("n_per_model","batch_size") if stage=="s5" else ("batch_size",))):
        raise ValueError("Seed must be >=0 and all counts/patience >0")
    if stage=="s4" and args.early_patience<=args.lr_patience:
        raise ValueError("Early patience must exceed LR patience")

def verified_s4_checkpoint(family,mode="formal"):
    folder=directory(family,mode)
    if not (folder/"best.keras").is_file() or not (folder/"metadata.json").is_file():
        raise FileNotFoundError(f"S4 checkpoint not found: {folder}")
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf8"))
    if (meta.get("input_sha256")!=EXPECTED_SHA
        or meta.get("max_trials")!=1342
        or meta.get("model_names")!=list(FAMILIES[family])
        or len(meta.get("features",[]))!=17
        or meta.get("feature_scale",[])[:2]!=[1.0,1.0]
        or meta.get("bayesflow")!="2.1.0"):
        raise ValueError(f"Checkpoint does not match frozen legacy design: {folder}")
    return folder

def run(stage,args):
    args_check(args,stage)
    info=validate_input()
    folder=directory(args.family,args.mode)
    print("[FROZEN INPUT]",json.dumps(info,indent=2),flush=True)
    print("[MODELS]",FAMILIES[args.family],flush=True)
    print("[OUTPUT]",folder,flush=True)
    if args.dry_run:
        return
    if stage=="s4":
        if (folder/"metadata.json").exists() or (folder/"best.keras").exists():
            raise FileExistsError(f"Refusing to overwrite formal experiment: {folder}")
        os.environ["ATT_BF_CLASSIFIER_LR_PATIENCE"]=str(args.lr_patience)
        os.environ["ATT_BF_CLASSIFIER_EARLY_PATIENCE"]=str(args.early_patience)
    else:
        verified_s4_checkpoint(args.family,args.mode)

    # Reuse original G2 matched-prior wrapper, BayesFlow trainer and S5/S6
    # inference without copying or revising the eight scientific models.
    from experiments import normalized_value_prior_sweep as sweep
    original_configure=sweep.configure_priors
    def configure_fixed(*values):
        if tuple(map(float,values))!=(1.,50.,.5,5.,0.,1.,0.,2.):
            raise ValueError("Legacy formal uses only frozen original G2 priors")
        original_configure(*values)
        sweep.OUTPUT=OUTPUT
    sweep.configure_priors=configure_fixed
    cli=["--stage",stage,"--family",args.family,"--mode",args.mode,
         "--input",str(INPUT),"--seed",str(args.seed)]
    if stage=="s4":
        cli += ["--epochs",str(args.epochs),"--batches",str(args.batches),
                "--batch-size",str(args.batch_size),"--validation",str(args.validation)]
        from experiments import parallel_normalized_value_prior_sweep as parallel
        call=parallel.main
        argv=["parallel_normalized_value_prior_sweep.py",
              "--sim-workers",str(args.sim_workers),*cli]
    else:
        call=sweep.main
        argv=["normalized_value_prior_sweep.py",*cli,"--batch-size",str(args.batch_size)]
        if stage=="s5":
            argv += ["--n-per-model",str(args.n_per_model)]
    old_argv=sys.argv
    try:
        sys.argv=argv
        call()
    finally:
        sys.argv=old_argv
    verified_s4_checkpoint(args.family,args.mode)
    if stage=="s4":
        meta=json.loads((folder/"metadata.json").read_text(encoding="utf8"))
        if (meta["config"]["num_batches_per_epoch"]!=args.batches
            or meta["config"]["batch_size"]!=args.batch_size
            or meta["early_stopping_patience"]!=args.early_patience):
            raise RuntimeError("Unexpected completed training configuration")
    print("[DONE]",stage,args.family,folder,flush=True)
