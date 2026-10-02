#!/usr/bin/env python
import argparse
from pathlib import Path

from data2param_flow import ParameterDecoder
from fit_att_mul_add.model_info import MODEL_INFOS, MODEL_ORDER

p = argparse.ArgumentParser(description="Train an 8-class amortized model-comparison classifier")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--val-split", type=float, default=0.10)
p.add_argument("--dropout", type=float, default=0.15)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--epochs", type=int, default=20000)
p.add_argument("--patience", type=int, default=30)
p.add_argument("--max-trial", type=int, default=20000)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

data_dirs = [str(Path("outputs/classifier_features") / m) for m in MODEL_ORDER]
for d in data_dirs:
    if not list(Path(d).glob("gen*.pkl")):
        raise FileNotFoundError(f"No classifier features in {d}")

param_names = MODEL_INFOS[MODEL_ORDER[-1]]["free_pnames"]

dec = ParameterDecoder(
    decoder_type="classify",
    dir_save="outputs/model_classifier",
    folder_save="all8",
    seed=a.seed,
)
dec.prepare_datafile(
    data_dirs,
    key_data_trial=["trialinfo", "choice", "rt", "fsmr"],
    key_param="params",
    param_names_use=param_names,
    batch_size=a.batch_size,
    val_split=a.val_split,
    seq_input_stratagy=1,
    max_trial=a.max_trial,
)
dec.prepare_model(lr=a.lr, post_dropout=a.dropout)
dec.train(a.epochs, a.patience)
dec.save_instance()
print("class order:", MODEL_ORDER)
