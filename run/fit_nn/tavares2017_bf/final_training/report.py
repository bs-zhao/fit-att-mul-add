"""Read-only report: compare final-data neural recovery to the old Tavares baseline.

No BayesFlow, GPU or training. Exposes S5 accuracy, per-class recovery,
training epochs, and S6 winning-model counts. Does NOT conflate
different held-out sizes: records n_recovery_per_model explicitly.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from final_training.plan import (OUTPUT, ROOT, INPUT, FAMILIES,
                                 classifier_dir, verify_final_input)

BASELINE = (ROOT / "outputs/tavares2017_bf/prior_sensitivity/"
 "normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0")


def baseline_folder(family):
    if family=="all":
        return BASELINE/"classifier/formal"
    return BASELINE/"classifier"/"__".join(FAMILIES[family])/"formal"


def read_csv(path):
    with path.open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))


def numeric(value):
    if value is None or value=="":
        return None
    v=float(value)
    return v if math.isfinite(v) else None


def data_from_folder(folder, family):
    p=folder/"diagnostics/metrics.json"
    meta=folder/"metadata.json"
    if not p.is_file() or not meta.is_file():
        return None
    m=json.loads(meta.read_text())
    values=json.loads(p.read_text())
    if m.get("model_names")!=list(FAMILIES[family]):
        raise ValueError(f"Bad model names for {folder}")
    perpath=folder/"diagnostics/per_model_metrics.csv"
    per=read_csv(perpath) if perpath.exists() else []
    predictions=folder/"diagnostics/predictions.csv"
    n_rec=None
    if predictions.exists():
        n_rec=len(read_csv(predictions))
    histpath=folder/"history.csv"
    hist=read_csv(histpath) if histpath.exists() else []
    loss=[numeric(r.get("val_loss")) for r in hist]
    loss=[v for v in loss if v is not None]
    s6=folder/"real_data/model_probabilities.csv"
    real=read_csv(s6) if s6.exists() else []
    winners=Counter(r["best_model"] for r in real) if real else Counter()
    real_subjects=len(real)
    if real and set(winners).difference(FAMILIES[family]):
        raise ValueError(f"Unexpected S6 model name in {s6}")
    return {
        "folder":str(folder),
        "n_recovery_trials_total":n_rec,
        "n_recovery_per_model":n_rec//len(FAMILIES[family]) if n_rec is not None
                               and n_rec%len(FAMILIES[family])==0 else None,
        "accuracy":numeric(values.get("accuracy")),
        "family_accuracy":numeric(values.get("family_accuracy")),
        "log_loss":numeric(values.get("log_loss")),
        "brier":numeric(values.get("brier")),
        "ece":numeric(values.get("ece")),
        "epochs_completed":len(hist),
        "best_validation_loss":min(loss) if loss else None,
        "n_real_subjects":real_subjects if real else None,
        "s6_winners":dict(winners),
        "per_model":per,
        "input_sha256":m.get("input_sha256"),
        "training_config":m.get("config"),
        "early_stopping_patience":m.get("early_stopping_patience"),
        "reduce_lr_patience":m.get("reduce_lr_patience"),
    }


def create_summary(include_baseline=True):
    source=verify_final_input()
    rows, per_rows, details={}, [], {}
    for family in ("ddm","race","all"):
        final_dir=classifier_dir(family)
        new=data_from_folder(final_dir,family)
        old=data_from_folder(baseline_folder(family),family) if include_baseline else None
        if new is not None and new["input_sha256"]!=source["input_sha256"]:
            raise ValueError(f"Final checkpoint was trained on different data: {final_dir}")
        if new is None:
            print("[SKIP] Incomplete final S5:",family,flush=True)
        details[family]={"new":new,"baseline":old}
        row={"family":family, "n_models":len(FAMILIES[family])}
        for prefix, info in (("new",new),("baseline",old)):
            for k in ("accuracy","family_accuracy","log_loss","brier","ece",
                      "epochs_completed","best_validation_loss",
                      "n_recovery_per_model","n_real_subjects",
                      "early_stopping_patience","reduce_lr_patience"):
                row[f"{prefix}_{k}"]=info.get(k) if info else None
            for name in FAMILIES[family]:
                row[f"{prefix}_winning_{name}"]=info["s6_winners"].get(name,0) if info and info["n_real_subjects"] is not None else None
        row["delta_recovery_accuracy"]=(new["accuracy"]-old["accuracy"]
            if new and old and new["accuracy"] is not None and old["accuracy"] is not None
            else None)
        rows[family]=row
        indexed_old={m["model"]:m for m in (old["per_model"] if old else [])}
        indexed_new={m["model"]:m for m in (new["per_model"] if new else [])}
        for name in FAMILIES[family]:
            n=indexed_new.get(name,{})
            o=indexed_old.get(name,{})
            rec_new=numeric(n.get("recall"))
            rec_old=numeric(o.get("recall"))
            per_rows.append({
                "family":family,"model":name,
                "new_recall":rec_new,"baseline_recall":rec_old,
                "delta_recall":rec_new-rec_old if rec_new is not None and rec_old is not None else None,
                "new_n":int(n["n"]) if "n" in n else None,
                "baseline_n":int(o["n"]) if "n" in o else None,
            })
    return source,list(rows.values()),per_rows,details


def save_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError("Nothing to save")
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-baseline",action="store_true")
    a=p.parse_args()
    source,rows,per,details=create_summary(include_baseline=not a.no_baseline)
    out=OUTPUT/"summary"
    out.mkdir(parents=True,exist_ok=True)
    save_rows(out/"model_recovery_comparison.csv",rows)
    save_rows(out/"per_model_recovery_comparison.csv",per)
    (out/"report.json").write_text(
        json.dumps({"final_input":source,"families":details},indent=2)+"\n",
        encoding="utf8")
    print("[SAVED]",out,flush=True)
    for r in rows:
        print("[RECOVERY]",r["family"],"old=",r["baseline_accuracy"],
              "new=",r["new_accuracy"],"delta=",r["delta_recovery_accuracy"],
              "n=",r["new_n_recovery_per_model"],flush=True)


if __name__=="__main__":
    main()
