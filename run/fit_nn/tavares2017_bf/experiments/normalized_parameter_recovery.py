"""Tavares2017 G2: S1 posterior training and S2 true parameter recovery.

This is *parameter* recovery, distinct from four-model S5 model recovery.
Both value normalization and parameter priors match the existing
normalized_datasetmax_v1 classifier experiment. No old checkpoints are used.

Usage:
    python -u experiments/normalized_parameter_recovery.py --stage s1 --model aDDM_t
    python -u experiments/normalized_parameter_recovery.py --stage s2 --model aDDM_t \
        --n-test 1000 --draws 1000
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["ATT_BF_VALUE_NORMALIZATION"] = "dataset_max"

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _shared import common
from _shared.data import model_infos, prior_spec
from experiments import normalized_value_prior_sweep as sweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("s1", "s2"), required=True)
    p.add_argument("--model", choices=common.MODEL_NAMES, required=True)
    p.add_argument("--mode", choices=("formal", "test"), default="formal")
    p.add_argument("--seed", type=common.nonnegative, default=20261030)
    p.add_argument("--input", type=Path,
                   default=common.ROOT / "data/tavares2017/trial_eye.csv")
    p.add_argument("--epochs", type=common.positive)
    p.add_argument("--batches", type=common.positive)
    p.add_argument("--batch-size", type=common.positive)
    p.add_argument("--validation", type=common.positive)
    p.add_argument("--n-test", type=common.positive)
    p.add_argument("--draws", type=common.positive)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if a.stage == "s1":
        if a.n_test is not None or a.draws is not None:
            p.error("--n-test/--draws are S2-only")
    else:
        if any(v is not None for v in (a.epochs, a.batches, a.validation)):
            p.error("--epochs/--batches/--validation are S1-only")
        if a.overwrite:
            p.error("--overwrite is S1-only")

    # Establish exactly the same prior, normalizer and input fingerprint as
    # the normalized-value four-classifier training experiment.
    sweep.NORMALIZATION_AUDIT = sweep.normalization_audit(a.input)
    sweep.configure_priors(1, 50, .5, 5, 0, 1, 0, 2)
    sweep.apply_shared_priors()

    folder = sweep.OUTPUT / a.model / a.mode
    spec = sweep.expected_spec([a.model], a.mode)
    spec.update(purpose="single_model_parameter_recovery", model=a.model)
    spec_file = folder / "prior_spec.json"
    if spec_file.exists():
        saved = json.loads(spec_file.read_text(encoding="utf8"))
        if saved != spec:
            raise ValueError(f"Existing parameter-recovery experiment differs: {spec_file}")
    elif a.stage == "s1":
        if (folder / "best.keras").exists():
            raise FileExistsError(f"Cannot verify pre-existing checkpoint: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        common.dump(spec_file, spec)
    else:
        raise FileNotFoundError(f"Run S1 first: missing {spec_file}")

    cli = ["--mode", a.mode, "--seed", str(a.seed),
           "--input", str(a.input), "--output", str(sweep.OUTPUT),
           "--model", a.model]
    if a.batch_size is not None:
        cli += ["--batch-size", str(a.batch_size)]
    elif a.stage == "s1":
        cli += ["--batch-size", "8"]
    else:
        cli += ["--batch-size", "4"]

    if a.stage == "s1":
        for flag, value in (("--epochs", a.epochs), ("--batches", a.batches),
                            ("--validation", a.validation)):
            if value is not None:
                cli += [flag, str(value)]
        if a.validation is None:
            cli += ["--validation", "64"]
        if a.overwrite:
            cli.append("--overwrite")
        opts = common.training_args(common.parser(
            "G2 normalized posterior training", model=True)).parse_args(cli)
        print("[S1] parameter posterior training", a.model, flush=True)
        print("[NORMALIZATION]", sweep.NORMALIZATION_AUDIT, flush=True)
        print("[PRIORS]", prior_spec([a.model]), flush=True)
        common.train(opts, classifier=False)
    else:
        # Unlike a four-model classifier, a single-model checkpoint's
        # metadata.model_names lists the entire catalog; verify its model
        # and actual free parameters instead.
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
        if metadata.get("model") != a.model:
            raise ValueError(f"Checkpoint is not {a.model}: {folder}")
        if metadata.get("parameter_names") != model_infos[a.model]["free_pnames"]:
            raise ValueError(f"Parameter names do not match: {folder}")
        if metadata.get("prior_spec") != prior_spec([a.model]):
            raise ValueError(f"Checkpoint parameter priors differ: {folder}")
        if metadata.get("feature_scale", [])[0:2] != [1.0, 1.0]:
            raise ValueError("Checkpoint values were not normalized before network input")
        if not (folder / "best.keras").is_file():
            raise FileNotFoundError(folder / "best.keras")
        if a.n_test is not None:
            cli += ["--n-test", str(a.n_test)]
        if a.draws is not None:
            cli += ["--draws", str(a.draws)]
        print("[S2] held-out parameter recovery and SBC", a.model, flush=True)
        # S2's load() verifies the exact prior_spec and input SHA256 again.
        sys.argv = ["s2_parameter_recovery.py", *cli]
        from s2_parameter_recovery import main as recover
        recover()
    print("[DONE]", a.stage, a.model, folder, flush=True)


if __name__ == "__main__":
    main()
