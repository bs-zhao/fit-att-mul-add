"""Rebuild the Tavares et al. (2017) Experiment 1 trial_eye.csv from original files.

This adapter preserves the *unchanged* c_bifood 8-model likelihood:
  simulator vs=[right,left], fix=1 left/0 right, choice=1 left/0 right.
Original angle deviations become larger-is-better similarity values 1..10.
Non-item gaze states (0,3 in the original data) are removed, NOT mapped to
left or right. Decision time becomes the sum of item-fixation durations, so
it is an *active-gaze decision clock*, not the raw button RT. The discarded
time and raw RT are preserved in the generated trial file and manifest.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = "https://raw.githubusercontent.com/glamlab/gaze-bias-differences/master/data/tavares_2017_fns/original/"
NAMES = ("expdata.csv", "fixations.csv")
FIELDS = (
    "subj", "trial", "v0", "v1", "response",
    "arr_ml3_left", "arr_ml3_time",
    "rt_raw_ms", "rt_active_ms", "nonitem_ms",
    "source_item_left", "source_item_right",
)


def similarity(angle):
    """Thomas et al. conversion: 0, 5, 10, 15 degrees -> 10, 7, 4, 1."""
    x = float(angle)
    if not math.isfinite(x) or abs(x) > 15:
        raise ValueError("Angle out of documented Tavares range: " + str(angle))
    return 10.0 - 0.6 * abs(x)


def fetch_sources(source_dir, cache_dir, refresh=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in NAMES:
        if source_dir is not None:
            payload = (source_dir / name).read_bytes()
        else:
            dest = cache_dir / name
            if not dest.exists() or refresh:
                with urllib.request.urlopen(SOURCE + name, timeout=120) as response:
                    payload = response.read()
                if not payload.startswith(b"parcode,"):
                    raise RuntimeError("Downloaded file does not look like the source CSV: " + name)
                temp = dest.with_suffix(".tmp")
                temp.write_bytes(payload)
                temp.replace(dest)
            payload = dest.read_bytes()
        result[name] = payload
    return result


def convert_data(trial_bytes, fixation_bytes):
    """Return model-ready records and audit with no dependency on BF/Pandas."""
    trials = list(csv.DictReader(io.StringIO(trial_bytes.decode("utf-8-sig"))))
    fixations = csv.DictReader(io.StringIO(fixation_bytes.decode("utf-8-sig")))
    required_trials = {"parcode", "trial", "rt", "choice", "item_left", "item_right"}
    required_fixes = {"parcode", "trial", "fix_item", "fix_time"}
    if not trials or not required_trials.issubset(trials[0]):
        raise ValueError("Unexpected expdata.csv fields")
    if not required_fixes.issubset(fixations.fieldnames or ()):
        raise ValueError("Unexpected fixations.csv fields")

    fixation_map = defaultdict(list)
    raw_fixations = 0
    for row in fixations:
        key = (int(row["parcode"]), int(row["trial"]))
        code = int(row["fix_item"])
        duration = float(row["fix_time"])
        if code not in (0, 1, 2, 3) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Bad fixation code/duration at " + str(key))
        if abs(duration - round(duration)) > 1e-6:
            raise ValueError("Sub-millisecond fixation duration at " + str(key))
        fixation_map[key].append((code, int(round(duration))))
        raw_fixations += 1

    records, rejected, present = [], Counter(), set()
    sum_raw, sum_active = 0, 0
    for row in trials:
        key = (int(row["parcode"]), int(row["trial"]))
        if key in present:
            raise ValueError("Duplicated trial key: " + str(key))
        present.add(key)
        if key not in fixation_map:
            raise ValueError("No eye data for " + str(key))
        events = fixation_map[key]
        raw_time = float(row["rt"])
        full_duration = sum(ms for _, ms in events)
        if not math.isfinite(raw_time) or abs(raw_time - full_duration) > 1.0:
            raise ValueError("Raw RT and complete fixation timeline differ at " + str(key))
        raw_choice = int(row["choice"])
        if raw_choice not in (-1, 1):
            raise ValueError("Unexpected raw choice at " + str(key))
        left_value = similarity(row["item_left"])
        right_value = similarity(row["item_right"])

        # Remove off-item gaze; then join consecutive equal *item* codes.
        # Crucially, fix_item=0 means off-item in Tavares but RIGHT in c_bifood.
        codes, lengths = [], []
        for code, ms in events:
            if code not in (1, 2) or ms <= 0:
                continue
            new_code = 1 if code == 1 else 0
            if codes and codes[-1] == new_code:
                lengths[-1] += ms
            else:
                codes.append(new_code)
                lengths.append(ms)
        active = sum(lengths)
        if active == 0:
            rejected["no_item_gaze"] += 1
            continue
        sum_raw += full_duration
        sum_active += active
        records.append({
            "subj": key[0], "trial": key[1],
            # c_bifood uses vs=[right,left], unlike Thomas' item_value_0=left
            "v0": right_value, "v1": left_value,
            "response": 1 if raw_choice == -1 else 0,
            "arr_ml3_left": repr(codes), "arr_ml3_time": repr(lengths),
            "rt_raw_ms": full_duration, "rt_active_ms": active,
            "nonitem_ms": full_duration - active,
            "source_item_left": float(row["item_left"]),
            "source_item_right": float(row["item_right"]),
        })
    if len(present) != len(fixation_map):
        raise ValueError("Fixations contain trial IDs absent from expdata.csv")
    if not records:
        raise ValueError("No analyzable trials")
    audit = {
        "original_trials": len(trials),
        "kept_trials": len(records),
        "excluded_trials": dict(rejected),
        "original_fixation_rows": raw_fixations,
        "subjects": len(set(r["subj"] for r in records)),
        "model_clock": "item_fixation_only",
        "mean_active_time_fraction": sum_active / sum_raw,
        "raw_rt_ms_total_kept": sum_raw,
        "active_rt_ms_total_kept": sum_active,
        "value_definition": "10 - 0.6*abs(raw_angular_deviation)",
        "value_order": "v0=right, v1=left",
        "response_order": "1=left, 0=right",
        "fixation_order": "1=left, 0=right",
        "not_full_button_rt": True,
    }
    return records, audit


def write_csv(records, path):
    buff = io.StringIO(newline="")
    writer = csv.DictWriter(buff, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    payload = buff.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, help="Directory containing original expdata.csv and fixations.csv; offline option")
    p.add_argument("--output", type=Path, default=ROOT / "data/tavares2017/trial_eye.csv")
    p.add_argument("--refresh", action="store_true", help="Redownload both upstream CSVs")
    a = p.parse_args()
    if a.refresh and a.source_dir is not None:
        p.error("--refresh is only available for downloaded sources")
    sources = fetch_sources(a.source_dir, ROOT / "data/tavares2017/original", refresh=a.refresh)
    records, audit = convert_data(sources["expdata.csv"], sources["fixations.csv"])
    audit["upstream"] = SOURCE
    audit["source_sha256"] = {name: hashlib.sha256(sources[name]).hexdigest() for name in NAMES}
    audit["prepared_csv_sha256"] = write_csv(records, a.output)
    audit["prepared_path"] = str(a.output.resolve())
    manifest = a.output.with_name("prep_manifest.json")
    manifest.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    print("[READY]", a.output, flush=True)


if __name__ == "__main__":
    main()
