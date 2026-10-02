#!/usr/bin/env python
import argparse
from pathlib import Path

from data2param_flow import ParameterDecoder
from fit_att_mul_add.model_info import MODEL_INFOS, MODEL_ORDER

p = argparse.ArgumentParser(description="Train conditional-flow parameter decoders")
p.add_argument("--model", choices=MODEL_ORDER + ["all"], default="all")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--val-split", type=float, default=0.06)
p.add_argument("--dropout", type=float, default=0.15)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--epochs", type=int, default=20000)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--max-trial", type=int, default=20000)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

models = MODEL_ORDER if a.model == "all" else [a.model]
for model in models:
    info = MODEL_INFOS[model]
    data_dir = Path("outputs/training_features") / model
    if not list(data_dir.glob("gen*.pkl")):
        raise FileNotFoundError(f"No training features in {data_dir}")
    decoder = ParameterDecoder(
        decoder_type="reg_point",
        dir_save="outputs/param_decoder",
        folder_save=model,
        seed=a.seed,
    )
    decoder.prepare_datafile(
        str(data_dir),
        key_data_trial=["trialinfo", "choice", "rt", "fsmr"],
        key_param="params",
        param_names_use=info["free_pnames"],
        batch_size=a.batch_size,
        val_split=a.val_split,
        seq_input_stratagy=1,
        max_trial=a.max_trial,
        ranges=info["free_pranges"],
    )
    decoder.prepare_model(lr=a.lr, post_dropout=a.dropout)
    decoder.train(a.epochs, a.patience)
    decoder.save_instance()
