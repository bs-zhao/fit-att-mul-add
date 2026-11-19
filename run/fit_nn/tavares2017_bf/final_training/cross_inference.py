"""Cross-inference ONLY: old/new pretrained classifiers on the opposite data.

This deliberately bypasses the original S6 'input hash == training hash'
restriction only for these specifically pinned and audited 2x2 comparisons.
NO checkpoint, training input, historical S6 output or model is modified.

From run/fit_nn/tavares2017_bf:
    python -u final_training/cross_inference.py --family ddm --checkpoint old --data new
    python -u final_training/cross_inference.py --family ddm --checkpoint new --data old
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["ATT_BF_VALUE_NORMALIZATION"] = "dataset_max"
HERE=Path(__file__).resolve().parent
BF_DIR=HERE.parent
sys.path.insert(0,str(BF_DIR))
from final_training.plan import (INPUT as NEW_DATA, OUTPUT as NEW_OUTPUT,
                                 ROOT, FAMILIES, classifier_dir, verify_final_input)

OLD_DATA=ROOT/"data/tavares2017/trial_eye.csv"
OLD_OUTPUT=(ROOT/"outputs/tavares2017_bf/prior_sensitivity/"
 "normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0")
CROSS_OUTPUT=NEW_OUTPUT/"cross_inference"
DATA_SHA={
    "old":"ff1479b10957aca922ec0792d87cb5bda0f6b19374d27c1eb6edc87c40bbf5be",
    "new":"dd3412fc49e38ec90de13f3efe71440f0d8fa2c5cf91b50ac8f66db255c5af04",
}
TRIALS={"old":31853,"new":30162}
MAX_TRIALS={"old":1342,"new":1339}
PRIOR_PROFILE="normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0"


def source_path(version):
    return OLD_DATA if version=="old" else NEW_DATA


def checkpoint_folder(version,family):
    if version=="new":
        return classifier_dir(family)
    if family=="all":
        return OLD_OUTPUT/"classifier/formal"
    return OLD_OUTPUT/"classifier"/"__".join(FAMILIES[family])/"formal"


def cell_dir(family,checkpoint,data):
    return CROSS_OUTPUT/"cells"/family/(checkpoint+"_classifier__"+data+"_data")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _all_prior_bounds(meta):
    prior=meta.get("prior_spec",{})
    if prior.get("profile")!=PRIOR_PROFILE:
        raise ValueError("Wrong historical G2 prior profile")
    bounds=prior.get("free_priors")
    if not isinstance(bounds,dict):
        raise ValueError("Missing generating parameter priors")
    return bounds


def validate_pair(family,checkpoint,data):
    """Check both genuine pretrained models and pinned data, no BF import."""
    if family not in FAMILIES or checkpoint not in DATA_SHA or data not in DATA_SHA:
        raise ValueError("Unknown family/checkpoint/data version")
    if checkpoint == data:
        raise ValueError("Diagonal S6 results exist already; only run opposite-data cells")
    final=verify_final_input()
    if final["input_sha256"]!=DATA_SHA["new"]:
        raise ValueError("Final dataset SHA changed since certified 2x2 design")
    for version in ("old","new"):
        csv_path=source_path(version)
        if not csv_path.exists() or sha256(csv_path)!=DATA_SHA[version]:
            raise ValueError(f"Missing or noncanonical {version} data: {csv_path}")
    metas={}
    for version in ("old","new"):
        folder=checkpoint_folder(version,family)
        best=folder/"best.keras"
        metadata=folder/"metadata.json"
        if not best.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"Checkpoint/metadata missing: {folder}")
        meta=json.loads(metadata.read_text(encoding="utf8"))
        if (meta.get("input_sha256")!=DATA_SHA[version]
            or meta.get("max_trials")!=MAX_TRIALS[version]
            or meta.get("model_names")!=list(FAMILIES[family])
            or len(meta.get("features",[]))!=17
            or list(meta.get("feature_scale",[])[:2])!=[1.0,1.0]):
            raise ValueError(f"Unexpected {version} checkpoint provenance: {folder}")
        norm=meta.get("prior_spec",{}).get("value_normalization",{})
        if (norm.get("method")!="dataset_max"
            or norm.get("input_sha256")!=DATA_SHA[version]
            or float(norm.get("source_value_max",0))!=10.0
            or norm.get("network_feature_scale")!=[1.0,1.0]):
            raise ValueError(f"Unexpected {version} normalization in {folder}")
        _all_prior_bounds(meta)
        metas[version]=meta
    if (metas["old"]["features"]!=metas["new"]["features"] or
        metas["old"]["feature_scale"]!=metas["new"]["feature_scale"] or
        _all_prior_bounds(metas["old"])!=_all_prior_bounds(metas["new"])):
        raise ValueError("Old and new classifier features or G2 priors differ")
    src=source_path(data)
    model_dir=checkpoint_folder(checkpoint,family)
    return {
        "family":family,"model_names":list(FAMILIES[family]),
        "checkpoint":checkpoint,"dataset":data,
        "checkpoint_path":str(model_dir/"best.keras"),
        "checkpoint_metadata":str(model_dir/"metadata.json"),
        "checkpoint_training_data_sha256":metas[checkpoint]["input_sha256"],
        "checkpoint_training_max_trials":metas[checkpoint]["max_trials"],
        "dataset_path":str(src),"dataset_sha256":DATA_SHA[data],
        "dataset_expected_max_trials":MAX_TRIALS[data],
        "output_directory":str(cell_dir(family,checkpoint,data)),
        "training_priors_identical":True,"feature_scale_identical":True,
        "note":"Input hash intentionally differs from pretrained checkpoint: this is controlled cross-inference, not ordinary S6.",
    }


def csv_bytes(rows,fields):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf8")


def write_atomic(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+".",
               suffix=".tmp",delete=False) as f:
            temporary=Path(f.name)
            f.write(payload)
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def infer(family,checkpoint,data,batch_size,seed):
    validation=validate_pair(family,checkpoint,data)
    dest=cell_dir(family,checkpoint,data)
    if (dest/"model_probabilities.csv").exists():
        raise FileExistsError(f"Already inferred; refusing to overwrite {dest}")
    import numpy as np
    import keras
    from _shared.common import environment
    from _shared.data import Experiment
    environment(seed)
    ex=Experiment(source_path(data))
    ids,raw,audit=ex.real()
    actual=int(np.sum(raw["summary_mask"]))
    if (ex.sha256!=DATA_SHA[data] or len(ids)!=25 or
        ex.max_trials!=MAX_TRIALS[data] or actual!=TRIALS[data] or
        len(audit)!=25 or
        any(a["input_trials"]!=a["retained_trials"] for a in audit)):
        raise ValueError(f"Noncanonical 17-feature real input: {data}, {actual} retained")
    names=list(FAMILIES[family]); k=len(names)
    model=keras.saving.load_model(validation["checkpoint_path"])
    blocks=[]
    for start in range(0,len(ids),batch_size):
        cond={key:value[start:start+batch_size] for key,value in raw.items()}
        try:
            estimates=model.estimate(conditions=cond,model_prior=np.full(k,1/k))
        except Exception as e:
            raise RuntimeError(
                "Cross-inference failed; checkpoint was trained on a different "
                f"padded-trial dimension ({validation['checkpoint_training_max_trials']} "
                f"vs input {ex.max_trials}). No trial was dropped or truncated."
            ) from e
        block=np.asarray(estimates["log_bayes_factors"])
        if block.shape!=(len(cond["trials"]),k) or not np.isfinite(block).all():
            raise ValueError("Invalid cross-inference classifier log Bayes factors")
        blocks.append(block)
    logbf=np.concatenate(blocks,axis=0)
    logit=logbf-logbf.max(axis=1,keepdims=True)
    prob=np.exp(logit)
    prob/=prob.sum(axis=1,keepdims=True)
    if not np.isfinite(prob).all() or not np.allclose(prob.sum(1),1.,atol=1e-6):
        raise ValueError("Cross-inference posterior probabilities invalid")
    best=np.asarray(names)[prob.argmax(1)]
    ddm=np.asarray([n.startswith("aDDM") for n in names])
    rows=[]
    for i,sid in enumerate(ids):
        q=prob[i]
        rows.append({
            "subj":sid,
            **{name:float(q[j]) for j,name in enumerate(names)},
            "best_model":best[i],
            "entropy":float(-np.sum(q*np.log(np.clip(q,1e-12,1)))),
            "p_DDM":float(q[ddm].sum()),
            "p_ACC":float(q[~ddm].sum()),
        })
    fields=["subj",*names,"best_model","entropy","p_DDM","p_ACC"]
    validation.update({
        "n_subjects":len(ids),"n_retained_trials":actual,
        "input_max_trials":ex.max_trials,"batch_size":batch_size,
        "seed":seed,"checkpoint_sha256":sha256(validation["checkpoint_path"]),
        "model_prior":[1/k]*k,
    })
    # Metadata last: a complete file is considered valid only with metadata.
    write_atomic(dest/"model_probabilities.csv",csv_bytes(rows,fields))
    write_atomic(dest/"real_data_trial_audit.csv",csv_bytes(audit,list(audit[0])))
    write_atomic(dest/"metadata.json",
        (json.dumps(validation,indent=2,ensure_ascii=False)+"\n").encode("utf8"))
    print("[DONE]",family,checkpoint,"checkpoint x",data,"dataset",
          "n=",len(ids),"Mul wins=",sum(m.endswith("_t") for m in best),
          "Add wins=",sum(m.endswith("_g") for m in best),flush=True)
    print("[OUTPUT]",dest,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family",choices=list(FAMILIES),required=True)
    p.add_argument("--checkpoint",choices=("old","new"),required=True)
    p.add_argument("--data",choices=("old","new"),required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--dry-run",action="store_true",
                   help="Check both checkpoint/data provenances without GPU/BayesFlow")
    args=p.parse_args()
    if args.batch_size<=0 or args.seed<0:
        p.error("Batch size must be positive, seed nonnegative")
    checks=validate_pair(args.family,args.checkpoint,args.data)
    print("[CROSS INFERENCE]",json.dumps(checks,indent=2),flush=True)
    if args.dry_run:
        print("[DRY-RUN] Verified pinned checkpoint/data; inference not run.")
        return
    infer(args.family,args.checkpoint,args.data,args.batch_size,args.seed)


if __name__=="__main__":
    main()
