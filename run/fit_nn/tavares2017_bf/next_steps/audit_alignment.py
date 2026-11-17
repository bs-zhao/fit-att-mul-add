"""Static model/network alignment and actual Tavares checkpoint metadata audit.

No BayesFlow required; checks source AST and saved S4 metadata.
Does not retrain or modify checkpoints.
"""
import argparse
import ast
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "run/fit_nn"


def definition(dataset):
    return runpy.run_path(str(BASE / dataset / "tools/model_info.py"))["model_infos"]


def ast_symbol(dataset, name, kind="function"):
    path = BASE / dataset / "_shared/common.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    is_class = kind == "class"
    for item in tree.body:
        if (isinstance(item, ast.ClassDef) if is_class else isinstance(item, ast.FunctionDef)) and item.name == name:
            return ast.dump(item, include_attributes=False)
    raise ValueError(f"{dataset}: function/class {name} missing")


def summary(output_root):
    food = definition("food_equal")
    kraj = definition("krajbich2010")
    target = ("aDDM_1","aDDM_2","aDDM_t","aDDM_g",
              "aRACE_1","aRACE_2","aRACE_t","aRACE_g")
    if set(food) != set(kraj) or set(food) != set(target):
        raise ValueError("Eight-model candidate catalog differs")
    fixed = {}
    for model in target:
        d1 = (food[model]["fixed_pnames"], food[model]["fixed_pvalues"])
        d2 = (kraj[model]["fixed_pnames"], kraj[model]["fixed_pvalues"])
        if d1 != d2:
            raise AssertionError(f"Fixed-parameter mismatch: {model}")
        if food[model]["free_pnames"] != kraj[model]["free_pnames"]:
            raise AssertionError(f"Free-parameter mismatch: {model}")
        fixed[model] = dict(zip(*d1))
    variants = ("food_equal_bf","krajbich2010_bf","tavares2017_bf",
                "tavares2017_bf_v2")
    symbols = [("workflow","function"),("adapter","function"),("environment","function"),
               ("AtomicBest","class"),("selected_models","function")]
    ast_equal = {}
    for name,kind in symbols:
        code = [ast_symbol(d,name,kind) for d in variants]
        ast_equal[name] = (len(set(code)) == 1)
    if not all(ast_equal.values()):
        raise AssertionError(f"Core architecture mismatch: {ast_equal}")
    family_names = {
        "ddm": target[:4], "race": target[4:], "all": target
    }
    checkpoints = {}
    specs = []
    for name, models in family_names.items():
        folder = (output_root / "classifier" /
                  ("__".join(models) if name != "all" else "") / "formal")
        path = folder / "metadata.json"
        if not path.exists():
            checkpoints[name] = {"present":False,"expected_path":str(path)}
            continue
        meta=json.loads(path.read_text())
        if meta.get("model_names") != list(models):
            raise AssertionError(f"Model names mismatch for {name}")
        if not (folder/"best.keras").exists():
            raise AssertionError(f"best.keras missing: {folder}")
        if meta.get("feature_scale", [])[:2] != [1.,1.]:
            raise AssertionError(f"Bad network value normalization: {name}")
        prior=meta.get("prior_spec")
        if not prior:
            raise AssertionError(f"Missing model prior spec: {name}")
        specs.append(prior)
        checkpoints[name] = {
            "present":True,
            "model_names":models,"config":meta.get("config"),
            "source_sha256":meta.get("input_sha256"),
            "max_trials":meta.get("max_trials"),
            "feature_scale":meta.get("feature_scale"),
            "best_keras_exists":True,
        }
    hashes = {r["source_sha256"] for r in checkpoints.values() if r.get("present")}
    if len(hashes)>1:
        raise AssertionError("Training inputs differ across family checkpoints")
    return {
        "model_catalog_identical":True,
        "fixed_parameters_identical":True,
        "fixed_by_model":fixed,
        "core_network_ast_identical":ast_equal,
        "tavares_checkpoints":checkpoints,
        "all_checkpoints_same_input":len(hashes)<=1,
        "training_config_note":"v1 batch_size 8 and validation 64, vs previous G2 256 and 2000; training hyperparameters are not fully matched",
        "value_and_timeline_note":"The baseline active clock removed off-item periods; full-timeline v2 is a separate model sensitivity, not an unchanged baseline.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-output",type=Path,default=ROOT /
       "outputs/tavares2017_bf/prior_sensitivity/"
       "normalized_datasetmax_v1_d1p0to50p0_a0p5to5p0_theta0p0to1p0_gamma0p0to2p0")
    p.add_argument("--output",type=Path,default=ROOT /
       "outputs/tavares2017_bf/next_steps/alignment.json")
    args=p.parse_args()
    info=summary(args.baseline_output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2),flush=True)
    print("[SAVED]",args.output)


if __name__=="__main__":
    main()
