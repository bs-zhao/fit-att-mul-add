"""Run final-data S4/S5/S6 WITHOUT forking or altering the established eight models.

Reuses the exact experiments.normalized_value_prior_sweep and its parallel
trainer. Only INPUT/OUTPUT paths, training simulation budget, validation
budget and patience are changed. S5/S6 always verify the saved checkpoint
against the FINAL CSV SHA256 and original matched G2 priors.

Examples:
 python -u final_training/run.py --stage s4 --family ddm
 python -u final_training/run.py --stage s5 --family ddm
 python -u final_training/run.py --stage s6 --family ddm
 python -u final_training/run.py --stage s4 --family ddm --dry-run
"""
import os
# Set these before any Keras/BayesFlow or _shared imports, including workers.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["ATT_BF_VALUE_NORMALIZATION"] = "dataset_max"
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BF_DIR = HERE.parent
sys.path.insert(0, str(BF_DIR))
from final_training.plan import (INPUT, OUTPUT, FAMILIES, classifier_dir,
                                 verify_final_input)


def positive(raw):
    n=int(raw)
    if n<1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("s4","s5","s6"),required=True)
    p.add_argument("--family",choices=("ddm","race","all"),required=True)
    p.add_argument("--mode",choices=("test","formal"),default="formal")
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--epochs",type=positive)
    p.add_argument("--batches",type=positive)
    p.add_argument("--batch-size",type=positive)
    p.add_argument("--validation",type=positive)
    p.add_argument("--sim-workers",type=positive)
    p.add_argument("--early-patience",type=positive)
    p.add_argument("--lr-patience",type=positive)
    p.add_argument("--n-per-model",type=positive)
    p.add_argument("--inference-batch",type=positive)
    p.add_argument("--dry-run",action="store_true",
                   help="Validate final input and print the exact isolated output; no BF import")
    args=p.parse_args(argv)
    if args.seed<0:p.error("--seed must be nonnegative")
    if args.stage!="s4" and any(getattr(args,name) is not None for name in
             ("epochs","batches","validation","sim_workers","early_patience","lr_patience")):
        p.error("Training hyperparameters only apply to --stage s4")
    if args.stage!="s5" and args.n_per_model is not None:
        p.error("--n-per-model applies only to S5")
    if args.stage=="s4" and args.inference_batch is not None:
        p.error("--inference-batch applies only to S5/S6")
    if args.stage!="s4" and args.batch_size is not None:
        p.error("--batch-size applies only to S4; use --inference-batch for S5/S6")
    if args.mode=="test":
        args.epochs=args.epochs or 2
        args.batches=args.batches or 2
        args.batch_size=args.batch_size or 2
        args.validation=args.validation or 8
        args.sim_workers=args.sim_workers or 2
        args.n_per_model=args.n_per_model or 8
    else:
        args.epochs=args.epochs or 5000
        args.batches=args.batches or 128
        args.batch_size=args.batch_size or 8
        args.validation=args.validation or 512
        args.sim_workers=args.sim_workers or 8
        args.n_per_model=args.n_per_model or 500
    args.inference_batch=args.inference_batch or 4
    args.early_patience=args.early_patience or 20
    args.lr_patience=args.lr_patience or 6
    if args.lr_patience >= args.early_patience:
        p.error("Require --lr-patience < --early-patience")
    return args


def plan(args):
    provenance=verify_final_input()
    folder=classifier_dir(args.family,args.mode)
    return {
        "study":"tavares2017_final_g2_train_v1",
        "stage":args.stage,"family":args.family,"models":FAMILIES[args.family],
        "mode":args.mode,"seed":args.seed,"checkpoint_dir":str(folder),
        "training_samples_per_epoch":args.batches*args.batch_size if args.stage=="s4" else None,
        "epochs":args.epochs if args.stage=="s4" else None,
        "batches":args.batches if args.stage=="s4" else None,
        "batch_size":args.batch_size if args.stage=="s4" else None,
        "validation_subjects":args.validation if args.stage=="s4" else None,
        "sim_workers":args.sim_workers if args.stage=="s4" else None,
        "early_stopping_patience":args.early_patience if args.stage=="s4" else None,
        "reduce_lr_patience":args.lr_patience if args.stage=="s4" else None,
        "recovery_subjects_per_model":args.n_per_model if args.stage=="s5" else None,
        "inference_batch":args.inference_batch if args.stage!="s4" else None,
        "provenance":provenance,
    }


def _patch_isolated_output(sweep):
    """Keep original sweeper's prior catalog and S4/S5/S6 code unchanged."""
    original=sweep.configure_priors

    def fixed_output(d0,d1,a0,a1,t0,t1,g0,g1):
        values=(float(d0),float(d1),float(a0),float(a1),
                float(t0),float(t1),float(g0),float(g1))
        if values!=(1.,50.,.5,5.,0.,1.,0.,2.):
            raise ValueError("Final training only supports the unchanged G2 prior bounds")
        original(*values)
        sweep.OUTPUT=OUTPUT
    sweep.configure_priors=fixed_output


def main(argv=None):
    args=parse_args(argv)
    p=plan(args)
    print("[FINAL STUDY]",json.dumps(p,indent=2),flush=True)
    if args.dry_run:
        print("[DRY-RUN] No model loaded, no output written.",flush=True)
        return
    checkpoint=classifier_dir(args.family,args.mode)
    if args.stage=="s4" and (checkpoint/"best.keras").exists():
        raise FileExistsError(
            f"Existing classifier would be overwritten: {checkpoint}/best.keras"
        )
    # Opt-in-only modification to the shared trainer; historical defaults
    # remain 10/4 unless these environment variables are set.
    if args.stage=="s4":
        os.environ["ATT_BF_CLASSIFIER_EARLY_PATIENCE"]=str(args.early_patience)
        os.environ["ATT_BF_CLASSIFIER_LR_PATIENCE"]=str(args.lr_patience)
    import experiments.normalized_value_prior_sweep as sweep
    _patch_isolated_output(sweep)
    cli=[
        "--stage",args.stage,
        "--family",args.family,
        "--mode",args.mode,
        "--input",str(INPUT.resolve()),
        "--seed",str(args.seed),
    ]
    if args.stage=="s4":
        cli += [
            "--epochs",str(args.epochs),
            "--batches",str(args.batches),
            "--batch-size",str(args.batch_size),
            "--validation",str(args.validation),
        ]
    elif args.stage=="s5":
        cli += ["--n-per-model",str(args.n_per_model),
                "--batch-size",str(args.inference_batch)]
    else:
        cli += ["--batch-size",str(args.inference_batch)]
    old_argv=sys.argv
    try:
        if args.stage=="s4":
            from experiments import parallel_normalized_value_prior_sweep as par
            sys.argv=["parallel_normalized_value_prior_sweep.py",
                      "--sim-workers",str(args.sim_workers),*cli]
            par.main()
        else:
            sys.argv=["normalized_value_prior_sweep.py",*cli]
            sweep.main()
    finally:
        sys.argv=old_argv
    if args.stage=="s4":
        meta=json.loads((checkpoint/"metadata.json").read_text(encoding="utf-8"))
        if (meta.get("input_sha256")!=p["provenance"]["input_sha256"]
            or meta.get("model_names")!=list(p["models"])
            or meta.get("early_stopping_patience")!=args.early_patience
            or meta.get("reduce_lr_patience")!=args.lr_patience):
            raise RuntimeError("Finished S4 checkpoint metadata does not match the final study")
        print("[VERIFIED FINAL S4]",checkpoint,flush=True)
    print("[COMPLETED]",args.stage,args.family,checkpoint,flush=True)


if __name__=="__main__":
    main()
