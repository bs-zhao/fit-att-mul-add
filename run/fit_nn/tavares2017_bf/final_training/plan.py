"""Immutable S4/S5/S6 study paths and canonical data guard (standard library only).

Call prepare_final.py first. Every stage verifies the EXACT final CSV and
its provenance. Output root is fixed: NEVER point to the old checkpoints.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BF_DIR = HERE.parent
ROOT = HERE.parents[3]
INPUT = ROOT / "data/tavares2017_final/trial_eye.csv"
OUTPUT = ROOT / "outputs/tavares2017_bf_final/g2_train_v1"
FAMILIES = {
    "ddm": ("aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g"),
    "race": ("aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g"),
}
FAMILIES["all"] = FAMILIES["ddm"] + FAMILIES["race"]
EXPECTED_SOURCE_HASHES = {
    "expdata.csv": "ee46db8a7fc94b84a36f64ff19d11a94a2da4717208b7daf4a194b0fe5a78ee7",
    "fixations.csv": "bbb8ca77502b6fa3c5a98ee6db6423dad95d625b3d5a82c88e4c1bb62da31e5e",
}
EXPECTED_FINAL_TRIALS = 30162


def classifier_dir(family, mode="formal"):
    if family not in FAMILIES:
        raise ValueError("Expected family ddm, race or all")
    if mode not in ("formal", "test"):
        raise ValueError("Expected formal or test")
    if family == "all":
        return OUTPUT / "classifier" / mode
    return OUTPUT / "classifier" / "__".join(FAMILIES[family]) / mode


def verify_final_input(input_path=INPUT, manifest_path=None):
    """Hard fail on unexpected/mismatched final data before GPU allocation."""
    p = Path(input_path).resolve()
    manifest_p = Path(manifest_path) if manifest_path else p.parent / "prep_manifest.json"
    if not p.is_file() or not manifest_p.is_file():
        raise FileNotFoundError(
            f"Final data/manifest missing: {p}. Run final_preprocessing/prepare_final.py"
        )
    meta = json.loads(manifest_p.read_text(encoding="utf-8"))
    blob = p.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != meta.get("prepared_sha256"):
        raise ValueError(f"Final CSV SHA256 differs from manifest: {digest}")
    if meta.get("source_sha256") != EXPECTED_SOURCE_HASHES:
        raise ValueError("Final input does not descend from the pinned original source")
    audit = meta.get("audited", {})
    bf = meta.get("bayesflow_feature_check", {})
    if (
        audit.get("original_trials") != 31854
        or audit.get("subjects") != 25
        or audit.get("final_trials") != EXPECTED_FINAL_TRIALS
        or audit.get("excluded_by_reason", {}).get("offitem_qc_gt50pct") != 1691
        or audit.get("excluded_by_reason", {}).get("feature_rt_limit") != 1
        or not bf.get("checked")
        or bf.get("retained_trials") != EXPECTED_FINAL_TRIALS
        or bf.get("feature_dimensions") != 17
        or bf.get("dataset_value_max") != 10
        or bf.get("input_sha256") != digest
    ):
        raise ValueError("Final preprocessing/QC/BayesFlow manifest is not certified")
    rows, ids, keys = 0, set(), set()
    with p.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        required = {
            "subj", "trial", "v0", "v1", "response",
            "arr_ml3_left", "arr_ml3_time",
            "rt_raw_ms", "rt_active_ms", "nonitem_ms",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Missing required columns")
        for r in reader:
            key = (int(r["subj"]), int(r["trial"]))
            if key in keys:
                raise ValueError(f"Duplicate participant/trial: {key}")
            keys.add(key)
            ids.add(key[0])
            if int(r["response"]) not in (0,1):
                raise ValueError(f"Invalid response at {key}")
            if not 1 <= float(r["v0"]) <= 10 or not 1 <= float(r["v1"]) <= 10:
                raise ValueError(f"Unexpected Thomas value at {key}")
            pos = ast.literal_eval(r["arr_ml3_left"])
            dur = ast.literal_eval(r["arr_ml3_time"])
            active_ms = int(r["rt_active_ms"])
            full_ms = int(r["rt_raw_ms"])
            off_ms = int(r["nonitem_ms"])
            if (not pos or len(pos)!=len(dur) or any(x not in (0,1) for x in pos)
                    or any(int(x)<=0 for x in dur) or sum(dur)!=active_ms
                    or full_ms != active_ms+off_ms or off_ms*2 > full_ms):
                raise ValueError(f"Final gaze/active clock/QC violation: {key}")
            rows += 1
    if rows != EXPECTED_FINAL_TRIALS or len(ids) != 25:
        raise ValueError(f"Wrong finalized participant/trial counts: {len(ids)}/{rows}")
    return {
        "input":str(p),
        "input_sha256":digest,
        "n_trials":rows,
        "n_subjects":len(ids),
        "source_sha256":meta["source_sha256"],
        "value_normalization":"dataset_max (Thomas values divided by 10 in original BF adapter)",
        "decision_clock":"option-fixations only (original binary Cython likelihood)",
        "output":str(OUTPUT),
        "thomas_reference_checked_locally":bool(
            meta.get("original_reproduced_to_thomas", {}).get("checked", False)
        ),
    }
