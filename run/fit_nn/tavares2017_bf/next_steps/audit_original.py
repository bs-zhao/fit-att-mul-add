"""Independent audit of original Tavares fixation data and the legacy active clock.

Usage: python -u next_steps/audit_original.py
Runs without BayesFlow or GPU. Produces per-trial and subject diagnostics.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/tavares2017/original")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tavares2017_bf/next_steps/data_audit")
    a = parser.parse_args()
    originals = {k: a.source_dir / k for k in ("expdata.csv", "fixations.csv")}
    for file in originals.values():
        if not file.is_file():
            parser.error(f"Missing {file}; run prepare_data.py first")
    events = defaultdict(list)
    with originals["fixations.csv"].open(newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["parcode"]), int(r["trial"]))
            events[key].append((int(r["fix_item"]), int(round(float(r["fix_time"])))))
    trial_rows, subject_rows = [], defaultdict(list)
    codes = Counter()
    with originals["expdata.csv"].open(newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["parcode"]), int(r["trial"]))
            fix = events[key]
            for code, ms in fix:
                codes[code] += ms
            total = sum(ms for _, ms in fix)
            active = sum(ms for code, ms in fix if code in (1, 2))
            off = total - active
            codes_kept = [code for code, ms in fix if code in (1, 2) and ms > 0]
            original_time = float(r["rt"])
            if abs(original_time - total) > 1.0:
                raise ValueError(f"Timing mismatch {key}: {original_time} vs {total}")
            item_left = 10 - .6 * abs(float(r["item_left"]))
            item_right = 10 - .6 * abs(float(r["item_right"]))
            choice_left = int(r["choice"]) == -1
            row = dict(
                subject=key[0], trial=key[1], raw_rt_ms=total,
                active_rt_ms=active, off_item_ms=off,
                off_item_fraction=off / total,
                high_off_item=int(2 * off > total),
                n_all_segments=len(fix),
                n_item_segments=len(codes_kept),
                n_item_runs=sum(j == 0 or v != codes_kept[j-1] for j, v in enumerate(codes_kept)),
                last_original_code=fix[-1][0],
                last_code_was_off_item=int(fix[-1][0] in (0,3)),
                values_tied=int(item_left == item_right),
                accuracy_on_unequal=int(choice_left == (item_left > item_right))
                    if item_left != item_right else "",
            )
            trial_rows.append(row)
            subject_rows[key[0]].append(row)
    a.output.mkdir(parents=True, exist_ok=True)
    save_csv(a.output / "trial_level.csv", trial_rows)
    subjects = []
    for subj, rows in sorted(subject_rows.items()):
        n = len(rows)
        subjects.append(dict(
            subject=subj, trials=n,
            high_off_item=sum(r["high_off_item"] for r in rows),
            excluded_percent=100 * sum(r["high_off_item"] for r in rows) / n,
            mean_off_item_fraction=sum(r["off_item_fraction"] for r in rows) / n,
            last_off_item=sum(r["last_code_was_off_item"] for r in rows),
            original_mean_rt_ms=sum(r["raw_rt_ms"] for r in rows) / n,
            active_mean_rt_ms=sum(r["active_rt_ms"] for r in rows) / n,
        ))
    save_csv(a.output / "subject_level.csv", subjects)
    total_ms = sum(codes.values())
    info = dict(
        input_sha256={name: hashlib.sha256(file.read_bytes()).hexdigest()
                      for name, file in originals.items()},
        n_trials=len(trial_rows), n_subjects=len(subjects),
        n_high_off_item=sum(x["high_off_item"] for x in trial_rows),
        n_last_off_item=sum(x["last_code_was_off_item"] for x in trial_rows),
        fraction_time_by_code={str(c): ms / total_ms for c,ms in sorted(codes.items())},
        fraction_time_active=(codes[1]+codes[2])/total_ms,
        note="QC > 50% off-item is trial exclusion. Active clock deletes all off-item times.",
    )
    (a.output / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)
    print("[SAVED]", a.output, flush=True)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
