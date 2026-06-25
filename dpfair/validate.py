import argparse
import torch

from dpfair.data import load_signed_dataset
from dpfair.metrics import delta_dpsp, evaluate_link_sign
from dpfair.run_utils import build_model, move_data, namespace_from_checkpoint_args, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = namespace_from_checkpoint_args(ckpt["args"])
    seed_everything(train_args.seed)

    data = load_signed_dataset(
        train_args.dataset,
        seed=train_args.seed,
        k=train_args.k,
        percentile=train_args.percentile,
        val_ratio=train_args.val_ratio,
        test_ratio=train_args.test_ratio,
        pape_hops=getattr(train_args, "pape_hops", 1),
        pape_max_paths=getattr(train_args, "pape_max_paths", 256),
    )
    data = move_data(data, device)
    model = build_model(train_args).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    features = ckpt["features"].to(device)
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
        perf = evaluate_link_sign(model, z, data.val_pos_edge_index, data.val_neg_edge_index)
        fair = delta_dpsp(model, z, data.val_pos_edge_index, data.val_neg_edge_index, data.head_mask)

    metrics = {**perf, **fair}
    print(
        "Validation AUC={auc:.4f} F1_POS={f1_pos:.4f} F1_NEG={f1_neg:.4f} "
        "MACRO_F1={macro_f1:.4f} ACC={acc:.4f} DeltaDPSP={delta_dpsp:.4f} "
        "HH_ACC={hh_acc:.4f} DOT_T_ACC={dot_t_acc:.4f} ACC_GAP={acc_gap:.4f}".format(**metrics)
    )


if __name__ == "__main__":
    main()
