"""One definitive, audited Tavares2017 Experiment-1 model-input dataset.

Model-invariant scientific choice: use the existing eight binary-eye Cython
models unchanged, so only left/right item-fixation time is modeled. Neither
off-item code 0 nor 3 is silently relabeled right gaze. Raw button RT is
retained as audit metadata and is NEVER substituted for the active RT.

Dataset rules are fixed, not configurable experimental alternatives:
  * Thomas et al. item values 10 - .6*abs(angular_deviation)
  * v0=right, v1=left; response 1=left, 0=right
  * exclude off-item gaze STRICTLY >50% of raw RT
  * apply exactly the current legacy 17-feature simulator representation limits
  * preserve both the full and active clocks for provenance

Writes a new, separate data/tavares2017_final tree. Baseline unchanged.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BF_DIR = HERE.parent
ROOT = BF_DIR.parents[2]
sys.path.insert(0, str(BF_DIR))
from prepare_data import (SOURCE, NAMES, FIELDS, convert_data,
                          fetch_sources, similarity)

SOURCE_SHA256 = {
    "expdata.csv": "ee46db8a7fc94b84a36f64ff19d11a94a2da4717208b7daf4a194b0fe5a78ee7",
    "fixations.csv": "bbb8ca77502b6fa3c5a98ee6db6423dad95d625b3d5a82c88e4c1bb62da31e5e",
}
THOMAS_URL = ("https://raw.githubusercontent.com/glamlab/"
              "gaze-bias-differences/master/data/tavares_2017_fns/"
              "tavares2017_prep_all.csv")
EXPECTED = {
    "original_trials": 31854, "original_fixations": 219994,
    "subjects": 25, "offitem_qc_exclusions": 1691,
    "feature_exclusions": 1, "final_trials": 30162,
}
# EXACT preexisting features() rule, not a new RT policy:
MAX_FIXATION_RUNS_EXCLUSIVE = 128
FEATURE_MAX_DURATION_MS = 28287
FEATURE_BINS_PER_3MS = FEATURE_MAX_DURATION_MS // 3


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _csv_bytes(rows, fields):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf8")


def _write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".",
                                         suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            f.write(payload)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def original_index(trial_bytes, eye_bytes):
    """Independent checks, not using the baseline converter's fixation logic."""
    trials = {}
    for row in csv.DictReader(io.StringIO(trial_bytes.decode("utf-8-sig"))):
        key = (int(row["parcode"]), int(row["trial"]))
        if key in trials:
            raise ValueError(f"Duplicate original trial: {key}")
        trials[key] = row
    eye = defaultdict(list)
    fix_count = 0
    for row in csv.DictReader(io.StringIO(eye_bytes.decode("utf-8-sig"))):
        key = (int(row["parcode"]), int(row["trial"]))
        code = int(row["fix_item"])
        duration = float(row["fix_time"])
        if code not in (0,1,2,3) or not math.isfinite(duration):
            raise ValueError(f"Invalid original fixation: {key}, {code}, {duration}")
        if duration < 0 or abs(round(duration)-duration)>1e-6:
            raise ValueError(f"Noninteger or negative fixation duration: {key}")
        eye[key].append((code, int(round(duration))))
        fix_count += 1
    if set(trials) != set(eye):
        raise ValueError("Trial/fixation keys differ")
    return trials, eye, fix_count


def independent_eye_encoding(original_eye):
    """Independently derive active gaze runs from the unmodified source."""
    active = []
    source_ms = Counter()
    for code, duration in original_eye:
        source_ms[code] += duration
        if code not in (1,2) or duration == 0:
            continue
        gaze = 1 if code == 1 else 0
        if active and active[-1][0] == gaze:
            active[-1][1] += duration
        else:
            active.append([gaze, duration])
    return active, source_ms


def feature_limit_reason(model_pos, model_dur):
    """Mirror the unchanged _shared.data.feature limits exactly."""
    if not model_pos or sum(model_dur) <= 0:
        return "no_item_gaze"
    if len(model_pos) >= MAX_FIXATION_RUNS_EXCLUSIVE:
        return "feature_fixation_count"
    if math.ceil(sum(model_dur) / 3) >= FEATURE_BINS_PER_3MS:
        return "feature_rt_limit"
    return ""


def assemble(trial_bytes, eye_bytes):
    original, eye, n_eye = original_index(trial_bytes, eye_bytes)
    prepared, baseline_audit = convert_data(trial_bytes, eye_bytes)
    lookup = {(int(r["subj"]),int(r["trial"])): r for r in prepared}
    if len(lookup)!=len(prepared):
        raise ValueError("Duplicate baseline prepared key")
    if set(lookup) != set(original):
        raise ValueError("Unexpected zero-option eye or missing trial in canonical source")

    audit = []
    retained = []
    reasons = Counter()
    source_durations = Counter()
    for key, source in original.items():
        rec = lookup[key]
        seq = eye[key]
        active_seq, counts = independent_eye_encoding(seq)
        source_durations.update(counts)
        full_ms = sum(counts.values())
        active_ms = counts[1]+counts[2]
        off_ms = counts[0]+counts[3]
        if abs(float(source["rt"])-full_ms)>1.0 or full_ms<=0:
            raise ValueError(f"Raw RT not equal to original fixation clock: {key}")
        if rec["rt_raw_ms"]!=full_ms or rec["rt_active_ms"]!=active_ms or rec["nonitem_ms"]!=off_ms:
            raise AssertionError(f"Prepared RT or off-item accounting mismatch: {key}")
        model_pos = ast.literal_eval(rec["arr_ml3_left"])
        model_dur = ast.literal_eval(rec["arr_ml3_time"])
        if model_pos != [p for p,_ in active_seq] or model_dur != [d for _,d in active_seq]:
            raise AssertionError(f"Incorrect model gaze recoding / merging: {key}")
        if sum(model_dur)!=active_ms or any(p not in (0,1) for p in model_pos):
            raise AssertionError(f"Non-binary gaze or clock violation: {key}")
        if rec["response"] != (1 if int(source["choice"])==-1 else 0):
            raise AssertionError(f"Choice L/R flip: {key}")
        if rec["v0"]!=similarity(source["item_right"]) or rec["v1"]!=similarity(source["item_left"]):
            raise AssertionError(f"Value L/R flip: {key}")

        off_fraction = off_ms/full_ms
        if 2*off_ms > full_ms:
            exclusion = "offitem_qc_gt50pct"
        else:
            exclusion = feature_limit_reason(model_pos,model_dur)
        if exclusion:
            reasons[exclusion] += 1
        else:
            retained.append(rec)
        audit.append({
            "subj":key[0],"trial":key[1],
            "included":int(not exclusion),"exclude_reason":exclusion,
            "raw_rt_ms":full_ms,"active_rt_ms":active_ms,
            "off_item_ms":off_ms,"off_item_fraction":off_fraction,
            "n_original_fixations":len(seq),"n_active_runs":len(model_pos),
            "first_original_code":seq[0][0],"last_original_code":seq[-1][0],
            "choice_original":int(source["choice"]),
            "choice_model":rec["response"],
            "value_left":rec["v1"],"value_right":rec["v0"],
        })

    if n_eye!=EXPECTED["original_fixations"] or len(original)!=EXPECTED["original_trials"]:
        raise AssertionError(f"Original counts changed: {len(original)} / {n_eye}")
    if len(set(x["subj"] for x in retained))!=EXPECTED["subjects"]:
        raise AssertionError("Unexpected participant count")
    if reasons["offitem_qc_gt50pct"]!=EXPECTED["offitem_qc_exclusions"]:
        raise AssertionError(f"QC count changed: {reasons}")
    if sum(v for k,v in reasons.items() if k!="offitem_qc_gt50pct")!=EXPECTED["feature_exclusions"]:
        raise AssertionError(f"Feature exclusion count changed: {reasons}")
    if len(retained)!=EXPECTED["final_trials"]:
        raise AssertionError(f"Final count changed: {len(retained)}")

    subjects = defaultdict(list)
    for row in audit:
        subjects[row["subj"]].append(row)
    subjects_csv = []
    for sid, trials in sorted(subjects.items()):
        reasons_for_subj = Counter(t["exclude_reason"] for t in trials)
        subjects_csv.append({
            "subj":sid, "original_trials":len(trials),
            "offitem_qc_excluded":reasons_for_subj["offitem_qc_gt50pct"],
            "feature_excluded":sum(v for k,v in reasons_for_subj.items()
                                   if k and k!="offitem_qc_gt50pct"),
            "final_trials":reasons_for_subj[""],
            "raw_rt_mean_ms":sum(t["raw_rt_ms"] for t in trials)/len(trials),
            "active_rt_mean_ms":sum(t["active_rt_ms"] for t in trials)/len(trials),
        })
    if sum(r["final_trials"] for r in subjects_csv)!=len(retained):
        raise AssertionError("Subject-level count mismatch")
    audit_summary = {
        "original_trials":len(original),
        "original_fixations":n_eye,
        "subjects":len(subjects_csv),
        "final_trials":len(retained),
        "excluded_by_reason":dict(reasons),
        "source_gaze_duration_ms":{str(k):v for k,v in sorted(source_durations.items())},
        "source_gaze_active_fraction":(
             (source_durations[1]+source_durations[2])/sum(source_durations.values())),
        "mean_original_raw_rt_ms":sum(x["raw_rt_ms"] for x in audit)/len(audit),
        "mean_original_active_rt_ms":sum(x["active_rt_ms"] for x in audit)/len(audit),
        "model_rt_definition":"sum of fixation durations only for original item codes 1 and 2",
        "model_fixation_codes":"1=left, 0=right; source codes 0/3 omitted",
        "choice":"model 1=left, 0=right; original -1=left, +1=right",
        "values":"v0=right, v1=left; 10-0.6*abs(angle); dataset-max /10 in BF",
        "original_button_rt_used_in_likelihood":False,
        "qc":"exclude when original off-item 0/3 time STRICTLY > 50% of original button RT",
        "feature_rules":{
            "max_fixation_runs_exclusive":MAX_FIXATION_RUNS_EXCLUSIVE,
            "legacy_max_duration_ms":FEATURE_MAX_DURATION_MS,
            "legacy_bins_per_3ms":FEATURE_BINS_PER_3MS,
        },
        "baseline_converter": {
            "original_trials":baseline_audit["original_trials"],
            "kept_trials":baseline_audit["kept_trials"],
        }
    }
    return retained, audit, subjects_csv, audit_summary, prepared


def validate_thomas_reference(source_prepared, reference_bytes):
    """Cross-check source-to-Thomas mapping for ALL original trials, pre-QC."""
    rows=list(csv.DictReader(io.StringIO(reference_bytes.decode("utf-8-sig"))))
    m={(int(r["subject"]),int(r["trial"])):r for r in rows}
    prep={(int(r["subj"]),int(r["trial"])):r for r in source_prepared}
    if len(m)!=len(rows) or set(m)!=set(prep):
        raise AssertionError("Thomas 2019 reference trial IDs are not 1-to-1")
    for key, r in prep.items():
        ref=m[key]
        pos=ast.literal_eval(r["arr_ml3_left"])
        dur=ast.literal_eval(r["arr_ml3_time"])
        left=sum(d for p,d in zip(pos,dur) if p==1)
        right=sum(d for p,d in zip(pos,dur) if p==0)
        if not all(math.isclose(a,float(ref[b]),abs_tol=1e-9,rel_tol=1e-12)
           for a,b in ((r["v1"],"item_value_0"),(r["v0"],"item_value_1"),
                       (r["rt_raw_ms"],"rt"),
                       (left/(left+right),"gaze_0"),(right/(left+right),"gaze_1"))):
            raise AssertionError(f"Thomas reference value/RT/gaze mismatch: {key}")
        if int(ref["choice"]) != 1-int(r["response"]):
            raise AssertionError(f"Thomas reference choice mismatch: {key}")
    return {"checked":True,"reference_rows":len(rows),
            "matched_original_trials":len(prep),"reference_sha256":sha(reference_bytes),
            "mismatch_count":0}


def validate_model_adapter(dataset):
    """Use actual final BayesFlow 17-feature adapter; no neural training needed."""
    os.environ["ATT_BF_VALUE_NORMALIZATION"]="dataset_max"
    from _shared.data import (Experiment, FEATURE_NAMES, SCALE, feature,
                              MAX_RT)
    if not math.isclose(float(MAX_RT), FEATURE_MAX_DURATION_MS/1000):
        raise AssertionError("Final preprocessing RT limit diverged from BF")
    if len(FEATURE_NAMES)!=17 or SCALE[:2].tolist()!=[1.,1.]:
        raise AssertionError("BF value feature normalization does not match final")
    ex=Experiment(dataset)
    if ex.value_max!=10.0 or len(ex.groups)!=25:
        raise AssertionError("Unexpected final BF value normalization or groups")
    ids, raw, subject_audit=ex.real()
    if len(ids)!=25 or any(x["input_trials"]!=x["retained_trials"] for x in subject_audit):
        raise AssertionError("BF S6 silently excluded trials from final-prepared CSV")
    if int(raw["summary_mask"].sum())!=EXPECTED["final_trials"]:
        raise AssertionError("BF 17-feature conversion changed trial counts")
    if raw["trials"].shape != (25,ex.max_trials,17):
        raise AssertionError("Final BF shape differs")
    return {
        "checked":True,"n_subjects":len(ids),
        "retained_trials":int(raw["summary_mask"].sum()),
        "max_trials":ex.max_trials,"feature_dimensions":17,
        "dataset_value_max":ex.value_max,
        "input_sha256":ex.sha256,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,
       help="Existing local expdata.csv/fixations.csv; default cached original")
    p.add_argument("--output",type=Path,
       default=ROOT/"data/tavares2017_final")
    p.add_argument("--fetch-thomas-reference",action="store_true",
       help="Download and verify every trial against Thomas 2019 public CSV")
    p.add_argument("--thomas-reference",type=Path,
       help="Offline Thomas 2019 tavares2017_prep_all.csv")
    p.add_argument("--refresh",action="store_true",
       help="Refresh original source files; only when not using --source-dir")
    args=p.parse_args()
    if args.source_dir and args.refresh:
        p.error("--refresh is not compatible with --source-dir")
    if args.fetch_thomas_reference and args.thomas_reference:
        p.error("Choose either --fetch-thomas-reference or --thomas-reference")
    raw=fetch_sources(args.source_dir, ROOT/"data/tavares2017/original",
                      refresh=args.refresh)
    actual_hash={name:sha(raw[name]) for name in NAMES}
    if actual_hash!=SOURCE_SHA256:
        raise ValueError(f"Original files differ from pinned canonical data: {actual_hash}")
    retained,audit,subjects,summary,all_prepared=assemble(
        raw["expdata.csv"],raw["fixations.csv"])
    if args.fetch_thomas_reference:
        with urllib.request.urlopen(THOMAS_URL,timeout=120) as response:
            reference=response.read()
    elif args.thomas_reference:
        reference=args.thomas_reference.read_bytes()
    else:
        reference=None
    reference_audit = (
        validate_thomas_reference(all_prepared, reference)
        if reference is not None else {"checked":False,
               "note":"Pass --fetch-thomas-reference or --thomas-reference for independent Thomas cross-check"}
    )
    dataset=args.output/"trial_eye.csv"
    dataset_bytes=_csv_bytes(retained,FIELDS)
    # Write the dataset before checking it against the real downstream adapter.
    _write_atomic(dataset,dataset_bytes)
    bf_check=validate_model_adapter(dataset)
    manifest={
        "version":"tavares2017_final_v1",
        "unique_final_dataset":True,
        "source":SOURCE,
        "source_sha256":actual_hash,
        "prepared_sha256":sha(dataset_bytes),
        "original_reproduced_to_thomas":reference_audit,
        "audited":summary,"bayesflow_feature_check":bf_check,
        "important":"The 8-model likelihood models active option-gaze time, NOT full button RT.",
        "baseline_data_untouched":True,
    }
    _write_atomic(args.output/"trial_audit.csv",_csv_bytes(audit, list(audit[0])))
    _write_atomic(args.output/"subject_audit.csv",_csv_bytes(subjects,list(subjects[0])))
    _write_atomic(args.output/"prep_manifest.json",
                  (json.dumps(manifest,indent=2,ensure_ascii=False)+"\n").encode("utf8"))
    print(json.dumps(manifest,indent=2,ensure_ascii=False),flush=True)
    print("[FINAL DATASET]",dataset,flush=True)
    print("[SHA256]",sha(dataset_bytes),flush=True)


if __name__=="__main__":
    main()
