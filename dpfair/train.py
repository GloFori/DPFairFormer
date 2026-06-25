import argparse
import time

import torch
from tqdm import trange

from dpfair.data import load_signed_dataset
from dpfair.metrics import delta_dpsp, evaluate_link_sign
from dpfair.run_utils import (
    append_csv,
    append_raw_results,
    build_model,
    checkpoint_payload,
    config_to_args,
    load_config,
    make_run_dir,
    move_data,
    save_json,
    seed_everything,
)


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default=None)
    config_args, _ = base_parser.parse_known_args()
    config_defaults = config_to_args(load_config(config_args.config))

    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("--dataset", default="Bitcoinotc")
    parser.add_argument("--model", default="dpfairsgnn", choices=["dpfairsgnn", "dpfairformer"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--pape_hops", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--pape_max_paths", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.08)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=5.0)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--transfer_weight", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ablation", default="full", choices=["full", "no_transfer", "no_fairness", "no_transformer"])
    parser.add_argument("--output_dir", default="dpfair/runs")
    parser.add_argument("--eval_metric", default="tradeoff", choices=["auc", "f1", "macro_f1", "acc", "delta_dpsp", "tradeoff"])
    parser.add_argument("--tradeoff_alpha", type=float, default=0.5)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def evaluate_split(model, features, data, split_name):
    pos_edges = getattr(data, f"{split_name}_pos_edge_index")
    neg_edges = getattr(data, f"{split_name}_neg_edge_index")
    with torch.no_grad():
        z = model(
            features,
            data.train_pos_edge_index,
            data.train_neg_edge_index,
            tail_mask=data.tail_mask,
            pos_degree=data.pos_degree,
            neg_degree=data.neg_degree,
            pos_pape=data.train_pos_pape,
            neg_pape=data.train_neg_pape,
        )
        perf = evaluate_link_sign(model, z, pos_edges, neg_edges)
        fair = delta_dpsp(model, z, pos_edges, neg_edges, data.head_mask)
    return {**perf, **fair}


def metric_score(metrics, eval_metric: str, tradeoff_alpha: float):
    if eval_metric == "delta_dpsp":
        return -metrics["delta_dpsp"]
    if eval_metric == "tradeoff":
        return metrics["auc"] - tradeoff_alpha * metrics["delta_dpsp"]
    return metrics[eval_metric]


def update_best_trackers(trackers, val_metrics, epoch_info, args, model, features, data, epoch, run_dir):
    scores = {
        "auc": val_metrics["auc"],
        "dpsp": -val_metrics["delta_dpsp"],
        "tradeoff": val_metrics["auc"] - args.tradeoff_alpha * val_metrics["delta_dpsp"],
        "selected": metric_score(val_metrics, args.eval_metric, args.tradeoff_alpha),
    }
    for name, score in scores.items():
        if score > trackers[name]["score"]:
            record = {
                **epoch_info,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "score": score,
                "threshold_k": data.threshold_k,
            }
            trackers[name] = record
            filename = "best.pt" if name == "selected" else f"best_{name}.pt"
            torch.save(checkpoint_payload(args, model, features, data, record, epoch), f"{run_dir}/{filename}")
    return trackers


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_dir = make_run_dir(args)
    save_json(f"{run_dir}/args.json", vars(args))

    data = load_signed_dataset(
        args.dataset,
        seed=args.seed,
        k=args.k,
        percentile=args.percentile,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pape_hops=args.pape_hops,
        pape_max_paths=args.pape_max_paths,
    )
    print(f"[Data] {data.name}: nodes={data.num_nodes}, train_pos={data.train_pos_edge_index.size(1)}, "
          f"train_neg={data.train_neg_edge_index.size(1)}, val_pos={data.val_pos_edge_index.size(1)}, "
          f"val_neg={data.val_neg_edge_index.size(1)}, test_pos={data.test_pos_edge_index.size(1)}, "
          f"test_neg={data.test_neg_edge_index.size(1)}, k={data.threshold_k}, "
          f"head={int(data.head_mask.sum())}, tail={int(data.tail_mask.sum())}")
    print(f"[Run] Logs and checkpoints: {run_dir}")

    model = build_model(args).to(device)

    features = model.create_spectral_features(data.train_pos_edge_index, data.train_neg_edge_index, data.num_nodes).to(device)
    data = move_data(data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trackers = {
        "auc": {"score": -float("inf"), "epoch": -1},
        "dpsp": {"score": -float("inf"), "epoch": -1},
        "tradeoff": {"score": -float("inf"), "epoch": -1},
        "selected": {"score": -float("inf"), "epoch": -1},
    }
    stale = 0
    start = time.time()
    for epoch in trange(args.epochs, desc="Training", ncols=100):
        model.train()
        optimizer.zero_grad()
        z, contexts = model(
            features,
            data.train_pos_edge_index,
            data.train_neg_edge_index,
            tail_mask=data.tail_mask,
            return_context=True,
            pos_degree=data.pos_degree,
            neg_degree=data.neg_degree,
            pos_pape=data.train_pos_pape,
            neg_pape=data.train_neg_pape,
        )
        mu = 0.0 if args.ablation == "no_fairness" else args.mu
        loss, parts = model.total_loss(
            z,
            contexts,
            data.train_pos_edge_index,
            data.train_neg_edge_index,
            data.head_mask,
            mu=mu,
            eta=args.eta,
        )
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.eval()
        val_metrics = evaluate_split(model, features, data, "val")
        epoch_info = {
            "epoch": epoch + 1,
            "loss": float(loss.detach()),
            **parts,
        }
        row = {
            **epoch_info,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_tradeoff": val_metrics["auc"] - args.tradeoff_alpha * val_metrics["delta_dpsp"],
        }
        append_csv(f"{run_dir}/metrics.csv", row)

        old_selected = trackers["selected"]["score"]
        trackers = update_best_trackers(
            trackers, val_metrics, epoch_info, args, model, features, data, epoch + 1, run_dir
        )
        if trackers["selected"]["score"] > old_selected:
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(f"\n[Early Stop] epoch={epoch + 1}, best_epoch={trackers['selected']['epoch']}")
            break

    test_by_checkpoint = {}
    for name, filename in (
        ("selected", "best.pt"),
        ("auc", "best_auc.pt"),
        ("dpsp", "best_dpsp.pt"),
        ("tradeoff", "best_tradeoff.pt"),
    ):
        payload = torch.load(f"{run_dir}/{filename}", map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        test_by_checkpoint[name] = evaluate_split(model, payload["features"].to(device), data, "test")
    summary = {
        "run_dir": run_dir,
        "time_seconds": time.time() - start,
        "best": trackers,
        "test": test_by_checkpoint,
    }
    save_json(f"{run_dir}/summary.json", summary)
    raw_results_path = append_raw_results(run_dir, args, trackers, test_by_checkpoint)

    print("\n[Training Finished]")
    print(f"Time: {time.time() - start:.2f}s")
    print(
        "Selected best val epoch={epoch} loss={loss:.4f} task={task:.4f} fair={fair:.4f} recon={recon:.4f} "
        "VAL_AUC={val_auc:.4f} VAL_F1_POS={val_f1_pos:.4f} VAL_F1_NEG={val_f1_neg:.4f} "
        "VAL_MACRO_F1={val_macro_f1:.4f} VAL_DeltaDPSP={val_delta_dpsp:.4f}".format(**trackers["selected"])
    )
    for name, test_metrics in test_by_checkpoint.items():
        print(
            f"[Test:{name}] AUC={test_metrics['auc']:.4f} F1_POS={test_metrics['f1_pos']:.4f} "
            f"F1_NEG={test_metrics['f1_neg']:.4f} MACRO_F1={test_metrics['macro_f1']:.4f} "
            f"ACC={test_metrics['acc']:.4f} DeltaDPSP={test_metrics['delta_dpsp']:.4f} "
            f"HH_ACC={test_metrics['hh_acc']:.4f} DOT_T_ACC={test_metrics['dot_t_acc']:.4f} "
            f"ACC_GAP={test_metrics['acc_gap']:.4f}"
        )
    print(f"Saved: {run_dir}/best.pt")
    print(f"Raw results: {raw_results_path}")


if __name__ == "__main__":
    main()
