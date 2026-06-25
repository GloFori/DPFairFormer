import csv
import json
import os
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .models import DPFairFormer, DPFairSGNN


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_data(data, device):
    for name, value in vars(data).items():
        if isinstance(value, torch.Tensor):
            setattr(data, name, value.to(device))
    return data


def make_run_dir(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{args.model}_{args.ablation}_seed{args.seed}_{stamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def flatten_config(config, prefix=""):
    flat = {}
    for key, value in config.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, name))
        else:
            flat[name] = value
    return flat


def load_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    return flatten_config(config or {})


def config_to_args(config):
    mapping = {
        "dataset_name": "dataset",
        "feature_dim": "features",
        "model_name": "model",
        "model_hidden_dim": "hidden",
        "model_num_layers": "layers",
        "model_encoder_layers": "encoder_layers",
        "model_dropout": "attention_dropout",
        "model_max_path_len": "pape_hops",
        "fairness_K_value": "k",
        "loss_mu": "mu",
        "loss_eta": "eta",
        "loss_lamb": "lamb",
        "train_epochs": "epochs",
        "train_lr": "lr",
        "train_weight_decay": "weight_decay",
        "train_grad_clip": "grad_clip",
        "train_patience": "patience",
        "train_seed": "seed",
        "eval_select_metric": "eval_metric",
        "eval_tradeoff_alpha": "tradeoff_alpha",
        "logging_output_dir": "output_dir",
    }
    args = {}
    for key, value in config.items():
        args[mapping.get(key, key)] = value
    return args


def namespace_from_checkpoint_args(args_dict):
    defaults = {
        "encoder_layers": 1,
        "attention_dropout": 0.1,
        "pape_hops": 1,
        "pape_max_paths": 256,
        "val_ratio": 0.08,
        "test_ratio": 0.2,
        "k": None,
        "percentile": None,
        "tradeoff_alpha": 0.5,
        "ablation": "full",
        "lamb": 5.0,
        "transfer_weight": 1.0,
    }
    merged = {**defaults, **args_dict}
    return SimpleNamespace(**merged)


def build_model(args):
    model_cls = DPFairFormer if args.model == "dpfairformer" and args.ablation != "no_transformer" else DPFairSGNN
    kwargs = dict(
        in_channels=args.features,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        lamb=args.lamb,
        transfer_weight=args.transfer_weight,
        use_transfer=args.ablation != "no_transfer",
    )
    if model_cls is DPFairFormer:
        kwargs["encoder_layers"] = getattr(args, "encoder_layers", 1)
        kwargs["attention_dropout"] = getattr(args, "attention_dropout", 0.1)
    return model_cls(**kwargs)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(args, model, features, data, best, epoch):
    return {
        "args": vars(args),
        "model_state": model.state_dict(),
        "features": features.detach().cpu(),
        "best": best,
        "epoch": epoch,
        "threshold_k": data.threshold_k,
    }


def append_raw_results(run_dir, args, trackers, test_by_checkpoint):
    table_path = os.path.join(args.output_dir, "tables", "raw_results.csv")
    for checkpoint_name, metrics in test_by_checkpoint.items():
        best = trackers.get(checkpoint_name, trackers["selected"])
        row = {
            "run_dir": run_dir,
            "dataset": args.dataset,
            "model": args.model,
            "ablation": args.ablation,
            "seed": args.seed,
            "checkpoint": checkpoint_name,
            "epoch": best.get("epoch", -1),
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "f1_pos": metrics["f1_pos"],
            "f1_neg": metrics["f1_neg"],
            "macro_f1": metrics["macro_f1"],
            "acc": metrics["acc"],
            "delta_dpsp": metrics["delta_dpsp"],
            "acc_gap": metrics["acc_gap"],
            "k": getattr(args, "k", None),
            "threshold_k": best.get("threshold_k", None),
        }
        append_csv(table_path, row)
    return table_path
