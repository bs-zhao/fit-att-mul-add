"""Generate six isolated Tavares2017 input profiles from ORIGINAL eye records.

Profiles independently vary: full vs active clock, Thomas vs zero-based
relative value scale, and the >50%-off-item trial QC rule.
Full clock preserves all intervals; fix_item 0/3 become neutral -1.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILES = {"active_thomas_all":{"clock":"active","values":"thomas","qc":False},"active_thomas_qc":{"clock":"active","values":"thomas","qc":True},"active_relative_qc":{"clock":"active","values":"relative","qc":True},"full_thomas_all":{"clock":"full","values":"thomas","qc":False},"full_thomas_qc":{"clock":"full","values":"thomas","qc":True},"full_relative_qc":{"clock":"full","values":"relative","qc":True}}
SOURCE_DIR = ROOT / "data/tavares2017/original"
DATA_ROOT = ROOT / "data/tavares2017_v2"
EXPECTED_SHA256 = {
    "expdata.csv": "ee46db8a7fc94b84a36f64ff19d11a94a2da4717208b7daf4a194b0fe5a78ee7",
    "fixations.csv": "bbb8ca77502b6fa3c5a98ee6db6423dad95d625b3d5a82c88e4c1bb62da31e5e",
}
FIELDS = (
    "subj", "trial", "v0", "v1", "response", "arr_ml3_left", "arr_ml3_time",
    "rt_raw_ms", "rt_active_ms", "nonitem_ms", "source_item_left",
    "source_item_right", "off_item_fraction",
)


def value(angle, scheme):
    x = abs(float(angle))
    if not math.isfinite(x) or x > 15.:
        raise ValueError(f"Invalid source angle: {angle}")
    if scheme == "thomas":
        return 10. - .6 * x
    if scheme == "relative":
        return 1. - x / 15.
    raise ValueError(scheme)


def convert(expdata_bytes, fixation_bytes, *, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    spec = PROFILES[profile]
    rows = list(csv.DictReader(io.StringIO(expdata_bytes.decode("utf-8-sig"))))
    fix_rows = csv.DictReader(io.StringIO(fixation_bytes.decode("utf-8-sig")))
    if not rows or not {"parcode", "trial", "choice", "item_left", "item_right", "rt"}.issubset(rows[0]):
        raise ValueError("Unexpected expdata fields")
    if not {"parcode", "trial", "fix_item", "fix_time"}.issubset(fix_rows.fieldnames or ()):
        raise ValueError("Unexpected fixations fields")
    events = defaultdict(list)
    for row in fix_rows:
        k = (int(row["parcode"]), int(row["trial"]))
        code, ms = int(row["fix_item"]), float(row["fix_time"])
        if code not in (0,1,2,3) or not math.isfinite(ms) or ms < 0 or abs(ms - round(ms)) > 1e-8:
            raise ValueError(f"Invalid eye data: {k}: {code} {ms}")
        events[k].append((code, int(round(ms))))

    out = []
    dropped = Counter()
    present = set()
    for row in rows:
        key = (int(row["parcode"]), int(row["trial"]))
        if key in present:
            raise ValueError(f"Repeated trial: {key}")
        present.add(key)
        if key not in events:
            raise ValueError(f"Missing fixations: {key}")
        seq = events[key]
        total = sum(ms for _,ms in seq)
        if not seq or abs(float(row["rt"]) - total) > 1:
            raise ValueError(f"RT mismatch {key}")
        off = sum(ms for c,ms in seq if c in (0,3))
        active = total - off
        if not active or not total:
            dropped["zero_active_or_raw_time"] += 1
            continue
        if spec["qc"] and 2 * off > total:
            dropped["off_item_above_50pct"] += 1
            continue
        response = int(row["choice"])
        if response not in (-1,1):
            raise ValueError("Unexpected choice")
        coded = []
        for code,ms in seq:
            if ms <= 0: continue
            if spec["clock"] == "active" and code not in (1,2): continue
            mapped = 1 if code == 1 else 0 if code == 2 else -1
            if spec["clock"] == "active" and coded and coded[-1][0] == mapped:
                coded[-1][1] += ms
            else:
                coded.append([mapped, ms])
        if not coded:
            dropped["no_intervals"] += 1
            continue
        dt = total if spec["clock"] == "full" else active
        assert sum(ms for _, ms in coded) == dt
        out.append({
            "subj": key[0], "trial": key[1],
            "v0": value(row["item_right"], spec["values"]),
            "v1": value(row["item_left"], spec["values"]),
            "response": 1 if response == -1 else 0,
            "arr_ml3_left": repr([c for c,_ in coded]),
            "arr_ml3_time": repr([ms for _,ms in coded]),
            "rt_raw_ms": total, "rt_active_ms": active, "nonitem_ms": off,
            "source_item_left": float(row["item_left"]),
            "source_item_right": float(row["item_right"]),
            "off_item_fraction": off / total,
        })
    if present != set(events):
        raise ValueError("Fixation file has unmatched trial keys")
    if not out:
        raise ValueError("No usable trials")
    manifest = {
        "profile": profile, "definition": spec,
        "original_trials": len(rows), "kept_trials": len(out),
        "excluded": dict(dropped), "n_subjects": len(set(x["subj"] for x in out)),
        "raw_rt_total_ms": sum(x["rt_raw_ms"] for x in out),
        "clock_rt_total_ms": sum(
            x["rt_raw_ms"] if spec["clock"] == "full" else x["rt_active_ms"] for x in out),
        "max_observed_rt_ms": max(
            x["rt_raw_ms"] if spec["clock"] == "full" else x["rt_active_ms"] for x in out),
        "gaze_encoding": "1=left 0=right -1=off_item (full only)",
        "value_encoding": "v0=right v1=left; normalization by dataset max in BayesFlow",
        "choice_encoding": "1=left 0=right",
    }
    return out, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=list(PROFILES)+["all"], default="all")
    p.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    p.add_argument("--output-root", type=Path, default=DATA_ROOT)
    args = p.parse_args()
    files = {name: (args.source_dir / name).read_bytes()
             for name in ("expdata.csv", "fixations.csv")}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    if hashes != EXPECTED_SHA256:
        raise ValueError(f"Original SHA256 mismatch: {hashes}, expected {EXPECTED_SHA256}")
    for profile in (list(PROFILES) if args.profile == "all" else [args.profile]):
        records, manifest = convert(files["expdata.csv"], files["fixations.csv"], profile=profile)
        folder = args.output_root / profile
        folder.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        payload = buf.getvalue().encode("utf8")
        (folder / "trial_eye.csv").write_bytes(payload)
        manifest["source_sha256"] = hashes
        manifest["prepared_sha256"] = hashlib.sha256(payload).hexdigest()
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("[READY]", profile, manifest["kept_trials"], folder / "trial_eye.csv", flush=True)


if __name__ == "__main__":
    main()
