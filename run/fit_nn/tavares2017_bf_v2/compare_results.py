"""Compare S5 model recovery and S6 real-subject evidence across v1/v2.

READ ONLY: never changes existing checkpoints or S6 model probability files.
Skip incomplete families/profiles, never invent missing results.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
EXPERIMENT="normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0"
MODELS={
    "ddm":["aDDM_1","aDDM_2","aDDM_t","aDDM_g"],
    "race":["aRACE_1","aRACE_2","aRACE_t","aRACE_g"],
}
MODELS["all"]=MODELS["ddm"]+MODELS["race"]


def folder(base,family):
    if family=="all":
        return base/"classifier/formal"
    return base/"classifier"/"__".join(MODELS[family])/"formal"


def load(base,family):
    f=folder(base,family)
    fname=f/"real_data/model_probabilities.csv"
    diag=f/"diagnostics/metrics.json"
    meta=f/"metadata.json"
    if not fname.is_file() or not meta.is_file():
        return None
    with fname.open(newline="",encoding="utf8") as h:
        rows=list(csv.DictReader(h))
    if not rows: raise RuntimeError(f"Empty S6 file: {fname}")
    expected=MODELS[family]
    for r in rows:
        for m in expected:
            if m not in r:raise RuntimeError(f"Missing {m} in {fname}")
    ids=[str(r["subj"]) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError(f"Duplicate real subject in {fname}")
    p=np.array([[float(row[m]) for m in expected] for row in rows])
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-4):
        raise ValueError(f"Bad normalized model probabilities: {fname}")
    metadata=json.loads(meta.read_text(encoding="utf8"))
    if metadata.get("model_names") != expected:
        raise ValueError(f"Model order mismatch: {f}")
    metrics=json.loads(diag.read_text(encoding="utf8")) if diag.exists() else {}
    return dict(
        subject_ids=ids, probs=p, names=expected, folder=f,
        meta=metadata, metrics=metrics,
    )


def global_summary(profile,family,entry):
    probs=entry["probs"]
    labels=entry["names"]
    best=probs.argmax(axis=1)
    winners=[labels[i] for i in best]
    out={
        "profile":profile,"family":family,"n_subjects":len(probs),
        "recovery_accuracy":entry["metrics"].get("accuracy",""),
        "recovery_family_accuracy":entry["metrics"].get("family_accuracy",""),
        "recovery_log_loss":entry["metrics"].get("log_loss",""),
        "recovery_ece":entry["metrics"].get("ece",""),
        "mean_max_pmp":float(np.max(probs,axis=1).mean()),
        "mean_entropy":float(np.mean(-np.sum(probs*np.log(np.maximum(probs,1e-12)),axis=1))),
        "training_batch_size":entry["meta"].get("config",{}).get("batch_size",""),
        "training_batches_per_epoch":entry["meta"].get("config",{}).get("num_batches_per_epoch",""),
        "input_sha256":entry["meta"].get("input_sha256",""),
    }
    for model in MODELS["all"]:
        out["mean_p__"+model]=float(probs[:,labels.index(model)].mean()) if model in labels else ""
        out["winners__"+model]=winners.count(model) if model in labels else ""
    if family=="all":
        ddm=np.sum(probs[:,:4],axis=1)
        mul=probs[:,2]+probs[:,6]
        add=probs[:,3]+probs[:,7]
        out.update(mean_p_DDM=float(ddm.mean()),
                   n_DDM_preferred=int((ddm>.5).sum()),
                   mean_p_Mul=float(mul.mean()),
                   mean_p_Add=float(add.mean()),
                   n_Add_gt_Mul=int((add>mul).sum()))
    return out


def paired(profile,ds):
    if not all(f in ds for f in ("ddm","race","all")):
        return []
    joint=ds["all"]
    rows=[]
    for fam,subset in (("ddm",MODELS["ddm"]),("race",MODELS["race"])):
        four=ds[fam]
        id_joint={id:i for i,id in enumerate(joint["subject_ids"])}
        id_four={id:i for i,id in enumerate(four["subject_ids"])}
        if set(id_joint)!=set(id_four):
            raise RuntimeError(f"Missing subjects in {profile}/{fam}")
        four_p=np.array([four["probs"][id_four[id]] for id in four["subject_ids"]])
        eight_p=np.array([joint["probs"][id_joint[id], [joint["names"].index(m) for m in subset]]
                          for id in four["subject_ids"]])
        mass=eight_p.sum(axis=1)
        if np.any(mass<=1e-12):
            raise RuntimeError("Near-zero family mass")
        conditional=eight_p/mass[:,None]
        for i,id in enumerate(four["subject_ids"]):
            rows.append({
                "profile":profile,"family":fam,"subject":id,
                "TV":float(.5*np.abs(four_p[i]-conditional[i]).sum()),
                "winner_agreement":int(np.argmax(four_p[i])==np.argmax(conditional[i])),
                "four_winner":subset[np.argmax(four_p[i])],
                "joint_conditional_winner":subset[np.argmax(conditional[i])],
                "joint_family_mass":float(mass[i]),
            })
    return rows


def comparisons(datas):
    rows=[]
    profiles=list(datas)
    for i,a in enumerate(profiles):
        for b in profiles[i+1:]:
            if (a=="baseline_v1" or b=="baseline_v1"):
                # Different feature scales / clocks: label as cross-version.
                source="cross_version_not_controlled"
            else:
                source="within_v2"
            for fam in ("ddm","race","all"):
                if fam not in datas[a] or fam not in datas[b]:
                    continue
                e1,e2=datas[a][fam],datas[b][fam]
                ids1={x:i for i,x in enumerate(e1["subject_ids"])}
                ids2={x:i for i,x in enumerate(e2["subject_ids"])}
                common=sorted(set(ids1)&set(ids2))
                if not common: continue
                p1=np.array([e1["probs"][ids1[x]] for x in common])
                p2=np.array([e2["probs"][ids2[x]] for x in common])
                rows.append({
                    "profile_A":a,"profile_B":b,"family":fam,"type":source,
                    "n_matched_subjects":len(common),
                    "mean_subject_TV":float(.5*np.abs(p1-p2).sum(axis=1).mean()),
                    "winner_agreement":float(np.mean(p1.argmax(1)==p2.argmax(1))),
                })
    return rows


def save_csv(path,rows):
    if not rows:return
    all_keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf8") as h:
        w=csv.DictWriter(h,fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles",nargs="*",help="Subset of V2 profiles; default all found")
    p.add_argument("--include-baseline",action="store_true")
    p.add_argument("--output",type=Path,
                   default=ROOT/"outputs/tavares2017_bf_v2/summary")
    a=p.parse_args()
    base=ROOT/"outputs/tavares2017_bf_v2"
    profiles=a.profiles or [p.name for p in base.iterdir() if p.is_dir()] if base.exists() else []
    if a.include_baseline:
        profiles=["baseline_v1"]+profiles
    result={}
    for profile in profiles:
        if profile=="baseline_v1":
            path=ROOT/"outputs/tavares2017_bf/prior_sensitivity"/EXPERIMENT
        else:
            path=base/profile/"prior_sensitivity"/EXPERIMENT
        ds={fam:entry for fam in MODELS if
            (entry:=load(path,fam)) is not None}
        if ds:result[profile]=ds
        else: print("[SKIP INCOMPLETE]",profile,flush=True)
    if not result:
        raise FileNotFoundError("No completed S6 model_probabilities.csv found")
    summary=[global_summary(profile,fam,d) for profile,entries in result.items()
             for fam,d in entries.items()]
    paired_rows=[r for profile,entries in result.items() for r in paired(profile,entries)]
    compared=comparisons(result)
    a.output.mkdir(parents=True,exist_ok=True)
    save_csv(a.output/"model_summary.csv",summary)
    save_csv(a.output/"four_vs_eight_real_consistency.csv",paired_rows)
    save_csv(a.output/"between_profile_differences.csv",compared)
    print("[DONE] Profiles:",list(result),flush=True)
    print("[SAVED]",a.output,flush=True)


if __name__=="__main__":
    main()
