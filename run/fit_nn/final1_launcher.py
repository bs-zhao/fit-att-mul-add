"""Final1 six-stage driver; run with the dataset-specific run.py entry point.

All six stages use the existing, tested 2.1.0 BayesFlow engine. The
per-process unified_g2 hook is switched to a separately versioned Final1
profile before any workflow function executes. Legacy files remain intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND","torch")
os.environ.setdefault("MPLBACKEND","Agg")
# The 'unified' Experiment path always performs global max normalization.
# This environment setting also protects any legacy helper imports.
os.environ["ATT_BF_VALUE_NORMALIZATION"]="dataset_max"

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from unified_final1 import (
    DATASETS,FAMILIES,MODELS,PRIORS,PROFILE,install,audit_experiment,
    config_hash,configuration,
)

STAGES={
    "s1":"s1_train_parameter_posterior.py",
    "s2":"s2_parameter_recovery.py",
    "s3":"s3_decode_real_data.py",
    "s4":"s4_train_model_classifier.py",
    "s5":"s5_model_recovery.py",
    "s6":"s6_classify_real_data.py",
}
def parse(argv=None):
    p=argparse.ArgumentParser(description="Three-dataset unified final1 S1-S6")
    p.add_argument("stage",choices=STAGES)
    p.add_argument("--dataset",choices=DATASETS,required=True)
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--family",choices=tuple(FAMILIES))
    p.add_argument("--mode",choices=("test","formal"),default="formal")
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--epochs",type=int)
    p.add_argument("--batches",type=int)
    p.add_argument("--batch-size",type=int)
    p.add_argument("--validation",type=int)
    p.add_argument("--n-test",type=int)
    p.add_argument("--draws",type=int)
    p.add_argument("--ppc-draws",type=int)
    p.add_argument("--n-per-model",type=int)
    p.add_argument("--overwrite",action="store_true",
                   help="Explicit S1/S4 retrain (never use to alter old experiments)")
    a=p.parse_args(argv)
    if a.seed<0:
        p.error("--seed must be nonnegative")
    if a.stage in ("s1","s2","s3"):
        if a.model is None or a.family is not None:
            p.error("S1-S3 require --model (not --family)")
    else:
        if a.family is None or a.model is not None:
            p.error("S4-S6 require --family (not --model)")
    if any(getattr(a,k) is not None and getattr(a,k)<1 for k in
           ("epochs","batches","batch_size","validation","n_test","draws","n_per_model")):
        p.error("All count settings must be >=1")
    if a.ppc_draws is not None and a.ppc_draws<0:
        p.error("--ppc-draws must be >=0")
    stage_allowed={
        "s1":{"epochs","batches","batch_size","validation","overwrite"},
        "s2":{"n_test","draws","batch_size"},
        "s3":{"draws","batch_size","ppc_draws"},
        "s4":{"epochs","batches","batch_size","validation","overwrite"},
        "s5":{"n_per_model","batch_size"},
        "s6":set(),
    }
    for k in ("epochs","batches","batch_size","validation","n_test","draws","ppc_draws","n_per_model"):
        if getattr(a,k) is not None and k not in stage_allowed[a.stage]:
            p.error(f"--{k.replace('_','-')} is not supported for {a.stage}")
    if a.overwrite and "overwrite" not in stage_allowed[a.stage]:
        p.error("--overwrite only allowed for S1/S4")
    return a

def folder(a):
    root=a.output or ROOT/"outputs"/f"{a.dataset}_final1"
    if a.stage in ("s1","s2","s3"):
        return root/a.model/a.mode
    if a.family=="all":
        return root/"classifier"/a.mode
    return root/"classifier"/"__".join(FAMILIES[a.family])/a.mode

def upstream_cli(a):
    """Map Final1 normalized data/priors to existing six stages exactly."""
    data=a.input or ROOT/"data"/a.dataset/"trial_eye.csv"
    output=a.output or ROOT/"outputs"/f"{a.dataset}_final1"
    cli=["--mode",a.mode,"--input",str(data),"--output",str(output),
         "--seed",str(a.seed)]
    if a.model is not None:
        cli.extend(("--model",a.model))
    else:
        cli.extend(("--models",*FAMILIES[a.family]))
    defaults={
        "s1":{"epochs":5000,"batches":50,"batch_size":64,"validation":2000},
        "s2":{"n_test":2000,"draws":1000,"batch_size":25},
        "s3":{"draws":2000,"batch_size":25,"ppc_draws":0},
        "s4":({"epochs":5000,"batches":128,"batch_size":8,"validation":512}
              if a.dataset=="tavares2017" else
              {"epochs":5000,"batches":50,"batch_size":128,"validation":2000}),
        "s5":{"n_per_model":2000,"batch_size":4 if a.dataset=="tavares2017" else 25},
        "s6":{},
    }[a.stage]
    for key,val in defaults.items():
        value=getattr(a,key)
        cli.extend(("--"+key.replace("_","-"),str(val if value is None else value)))
    if a.overwrite:
        cli.append("--overwrite")
    return cli

def main(argv=None):
    a=parse(argv)
    upstream=HERE/f"{a.dataset}_bf"
    target=upstream/STAGES[a.stage]
    if not target.is_file():
        raise FileNotFoundError(f"Missing original stage implementation: {target}")
    data=a.input or ROOT/"data"/a.dataset/"trial_eye.csv"
    if not data.is_file():
        raise FileNotFoundError(f"Missing prepared trial_eye.csv: {data}")
    if str(upstream) not in sys.path:
        sys.path.insert(0,str(upstream))
    # Important: _shared.data loads the same g2 instance into _shared.common.
    # The Final1 patch occurs *before* common/workflow imports.
    source=importlib.import_module("_shared.data")
    models=install(source.unified_g2)
    from _shared import common
    if common.unified_g2 is not source.unified_g2:
        raise RuntimeError("G2 module alias mismatch; refusing to run")
    ex=source.Experiment(data,unified=True)
    audit=audit_experiment(ex)
    output=folder(a)
    manifest={
        "experiment":f"{a.dataset}_final1",
        "dataset":a.dataset,
        "stage":a.stage,
        "selected_models":[a.model] if a.model else list(FAMILIES[a.family]),
        "input":str(data),
        "output":str(output),
        "unified_config":configuration(),
        "normalization_and_data":audit,
        "cli":upstream_cli(a),
    }
    print("[FINAL1 PREFLIGHT]",json.dumps(manifest,indent=2),flush=True)
    if a.dry_run:
        print("[DRY RUN] Verified unified 8 models, fixed/free parameters, priors, source values",flush=True)
        return
    if a.stage in ("s2","s3","s5","s6"):
        # Verify all checkpoint metadata before S2/S3/S5/S6, not only after loading.
        metadata_file=output/"metadata.json"
        if not (output/"best.keras").is_file():
            raise FileNotFoundError(f"Missing preceding training checkpoint {output/'best.keras'}")
        metadata=json.loads(metadata_file.read_text(encoding="utf8"))
        expected_names=[a.model] if a.model else list(FAMILIES[a.family])
        source.unified_g2.validate_checkpoint(metadata,ex,expected_names,
                                               source.FEATURE_NAMES,output)
    else:
        if (output/"best.keras").exists() and not a.overwrite:
            raise FileExistsError(f"Checkpoint already exists: {output}; use --overwrite")
        output.mkdir(parents=True,exist_ok=True)
        manifest_path=output/"final1_manifest.json"
        if manifest_path.is_file():
            existing=json.loads(manifest_path.read_text(encoding="utf8"))
            for key in ("experiment","dataset","selected_models","input",
                        "unified_config","normalization_and_data"):
                if existing[key]!=manifest[key]:
                    raise ValueError(f"Existing Final1 manifest mismatch in {key}: {manifest_path}")
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")

    # Tavares classifiers: ensure patience allows learning-rate reductions.
    if a.stage=="s4" and a.dataset=="tavares2017":
        os.environ.setdefault("ATT_BF_CLASSIFIER_LR_PATIENCE","8")
        os.environ.setdefault("ATT_BF_CLASSIFIER_EARLY_PATIENCE","30")
    previous=sys.argv
    try:
        sys.argv=[str(target),*upstream_cli(a)]
        runpy.run_path(str(target),run_name="__main__")
    finally:
        sys.argv=previous
    print("[DONE FINAL1]",a.dataset,a.stage,a.model or a.family,output,flush=True)

if __name__=="__main__":
    main()
