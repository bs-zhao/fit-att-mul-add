"""Summarize exact 2x2 cross-inference with old/new pretrained G2 models.

Uses existing diagonal S6 outputs, fresh opposite-data inference outputs.
All model probabilities are conditional on the SAME named candidate family.
Adds a fully decomposed Mul winner/mean-posterior comparison, including
DDM-Mul minus RACE-Mul, without selecting a preferred outcome.
"""
import argparse
import csv
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from final_training.cross_inference import (
    CROSS_OUTPUT, DATA_SHA, TRIALS, FAMILIES, checkpoint_folder, cell_dir,
    write_atomic, csv_bytes
)

FOUR_CELLS=(("old","old"),("old","new"),("new","old"),("new","new"))


def read_csv(path):
    with Path(path).open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))


def cell_file(family,checkpoint,data):
    if checkpoint==data:
        return checkpoint_folder(checkpoint,family)/"real_data/model_probabilities.csv"
    return cell_dir(family,checkpoint,data)/"model_probabilities.csv"


def verified_cell(family,checkpoint,data):
    path=cell_file(family,checkpoint,data)
    if not path.is_file():
        raise FileNotFoundError(f"Missing S6 cell {family}/{checkpoint}/{data}: {path}")
    manifest_path=(checkpoint_folder(checkpoint,family)/"metadata.json"
        if checkpoint==data else cell_dir(family,checkpoint,data)/"metadata.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint/cross-inference provenance: {manifest_path}")
    provenance=json.loads(manifest_path.read_text(encoding="utf8"))
    if checkpoint==data:
        if provenance.get("input_sha256")!=DATA_SHA[data]:
            raise ValueError(f"Diagonal checkpoint input SHA differs: {path}")
    else:
        if (provenance.get("dataset_sha256")!=DATA_SHA[data]
            or provenance.get("checkpoint_training_data_sha256")!=DATA_SHA[checkpoint]
            or provenance.get("family")!=family
            or provenance.get("checkpoint")!=checkpoint
            or provenance.get("dataset")!=data
            or provenance.get("n_subjects")!=25
            or provenance.get("n_retained_trials")!=TRIALS[data]):
            raise ValueError(f"Incorrect crossed S6 provenance: {manifest_path}")
    names=list(FAMILIES[family])
    rows=read_csv(path)
    subjects={}
    for row in rows:
        sid=str(row.get("subj"))
        if sid in subjects:
            raise ValueError(f"Duplicate subject in {path}: {sid}")
        prob=[float(row[name]) for name in names]
        if (not all(math.isfinite(v) and -1e-8 <= v <= 1+1e-8 for v in prob)
            or abs(sum(prob)-1.)>1e-4):
            raise ValueError(f"Invalid model posterior in {path}: {sid}")
        argmax=max(range(len(names)),key=lambda i:prob[i])
        predicted=names[argmax]
        if row.get("best_model")!=predicted:
            raise ValueError(f"Best-model field doesn't match posterior: {path} subj {sid}")
        subjects[sid]={"probs":dict(zip(names,prob)),"best":predicted}
    if len(subjects)!=25:
        raise ValueError(f"Expected 25 subjects, got {len(subjects)} in {path}")
    return subjects, str(path)


def summed(sid_entry,suffix):
    return sum(value for model,value in sid_entry["probs"].items()
               if model.endswith(suffix))


def summaries(families):
    """Pure read-only synthesis, with no P-hacking selection."""
    cell_stats=[]
    subject_rows=[]
    fourway={}
    provenance={}
    for family in families:
        cells={}
        sources={}
        for ck,dat in FOUR_CELLS:
            key=ck+"_"+dat
            cells[key],sources[key]=verified_cell(family,ck,dat)
        base_ids=set(cells["old_old"])
        for key,records in cells.items():
            if set(records)!=base_ids:
                raise ValueError(f"Mismatched subjects in {family}/{key}")
        provenance[family]=sources
        stat={}
        names=FAMILIES[family]
        for ck,dat in FOUR_CELLS:
            key=ck+"_"+dat
            records=cells[key]
            predicted=[records[sid]["best"] for sid in sorted(base_ids)]
            wins=Counter(predicted)
            mean={n:sum(r["probs"][n] for r in records.values())/len(records)
                  for n in names}
            mul_wins=sum(wins[n] for n in names if n.endswith("_t"))
            add_wins=sum(wins[n] for n in names if n.endswith("_g"))
            mean_mul=sum(mean[n] for n in names if n.endswith("_t"))
            mean_add=sum(mean[n] for n in names if n.endswith("_g"))
            stat[key]={"mul_winners":mul_wins,"add_winners":add_wins,
                       "mean_p_mul":mean_mul,"mean_p_add":mean_add}
            cell_stats.append({
                "family":family,"classifier":ck,"data":dat,
                "n_subjects":len(records),"mul_winners":mul_wins,
                "add_winners":add_wins,"mean_p_mul":mean_mul,
                "mean_p_add":mean_add,
                **{"wins_"+n:wins[n] for n in names},
                **{"mean_p_"+n:mean[n] for n in names},
            })
            for sid in sorted(base_ids,key=lambda s:int(s)):
                r=records[sid]
                subject_rows.append({
                    "family":family,"subj":sid,
                    "classifier":ck,"data":dat,"best_model":r["best"],
                    "p_mul":summed(r,"_t"),"p_add":summed(r,"_g"),
                    **{"p_"+n:r["probs"][n] for n in names},
                })
        fourway[family]=stat
    return cell_stats,subject_rows,fourway,provenance


def factorial_decomposition(fourway):
    """2×2 effect decomposition, descriptive not causal identification.

    Cells: A old/old; B old/new; C new/old; D new/new.
    Data contrast at old checkpoint B-A, at new checkpoint D-C.
    Training contrast at old data C-A, at new data D-B.
    Interaction D-C-B+A.
    """
    out=[]
    for family,group in fourway.items():
        for metric in ("mul_winners","add_winners","mean_p_mul","mean_p_add"):
            A=group["old_old"][metric]
            B=group["old_new"][metric]
            C=group["new_old"][metric]
            D=group["new_new"][metric]
            out.append({
                "family":family,"metric":metric,
                "old_model_old_data":A,
                "old_model_new_data":B,
                "new_model_old_data":C,
                "new_model_new_data":D,
                "data_effect_at_old_classifier_B_minus_A":B-A,
                "data_effect_at_new_classifier_D_minus_C":D-C,
                "classifier_effect_on_old_data_C_minus_A":C-A,
                "classifier_effect_on_new_data_D_minus_B":D-B,
                "interaction_D_minus_C_minus_B_plus_A":D-C-B+A,
                "total_change_D_minus_A":D-A,
            })
    return out


def ddm_race_comparison(fourway):
    if not all(f in fourway for f in ("ddm","race")):
        return []
    rows=[]
    for ck,dat in FOUR_CELLS:
        key=ck+"_"+dat
        d=fourway["ddm"][key];r=fourway["race"][key]
        rows.append({
            "classifier":ck,"data":dat,
            "ddm_mul_winners":d["mul_winners"],
            "race_mul_winners":r["mul_winners"],
            "ddm_minus_race_mul_winners":d["mul_winners"]-r["mul_winners"],
            "ddm_mean_p_mul":d["mean_p_mul"],
            "race_mean_p_mul":r["mean_p_mul"],
            "ddm_minus_race_mean_p_mul":d["mean_p_mul"]-r["mean_p_mul"],
            "ddm_add_winners":d["add_winners"],
            "race_add_winners":r["add_winners"],
        })
    return rows


def write_rows(path,rows):
    if not rows:
        return
    headers=list(dict.fromkeys(key for row in rows for key in row))
    write_atomic(path,csv_bytes(rows,headers))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families",nargs="+",choices=list(FAMILIES),
                   default=["ddm","race"],
                   help="Default both four-model families; optional all")
    args=p.parse_args()
    families=list(dict.fromkeys(args.families))
    cells,subject,table,sources=summaries(families)
    decomposition=factorial_decomposition(table)
    comparison=ddm_race_comparison(table)
    dest=CROSS_OUTPUT/"summary"
    write_rows(dest/"four_cells.csv",cells)
    write_rows(dest/"per_subject.csv",subject)
    write_rows(dest/"two_by_two_decomposition.csv",decomposition)
    if comparison:
        write_rows(dest/"ddm_vs_race_mul.csv",comparison)
    provenance={"families":families,"cell_files":sources,
                "input_sha256":DATA_SHA,
                "note":"Descriptive 2x2 contrasts. Not controlled retraining or a causal decomposition of data vs training; network approximations may interact with QC."}
    write_atomic(dest/"provenance.json",
        (json.dumps(provenance,indent=2)+"\n").encode("utf8"))
    print("[SAVED]",dest,flush=True)
    print("[TWO BY TWO / MUL DDM-RACE]",flush=True)
    for r in comparison:print(r,flush=True)
    for r in decomposition:
        if r["metric"]=="mul_winners":
            print("[MUL EFFECT]",r,flush=True)


if __name__=="__main__":
    main()
