from typing import Dict, Tuple

import torch
from sklearn.metrics import f1_score, roc_auc_score


def edge_logits(model, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
    pos_logits = model.discriminate(z, pos_edge_index)[:, :2]
    neg_logits = model.discriminate(z, neg_edge_index)[:, :2]
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([
        torch.zeros(pos_logits.size(0), dtype=torch.long, device=z.device),
        torch.ones(neg_logits.size(0), dtype=torch.long, device=z.device),
    ])
    return logits, labels


def evaluate_link_sign(model, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> Dict[str, float]:
    logits, labels = edge_logits(model, z, pos_edge_index, neg_edge_index)
    pred = logits.argmax(dim=1)
    neg_prob = torch.softmax(logits, dim=1)[:, 1]
    y_true = labels.detach().cpu().numpy()
    y_score = neg_prob.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)),
        "f1_neg": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acc": float((pred == labels).float().mean().item()),
        "pred_pos_rate": float(pred.eq(0).float().mean().item()),
        "pred_neg_rate": float(pred.eq(1).float().mean().item()),
    }


def _edge_group_masks(edge_index: torch.Tensor, head_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    src, dst = edge_index.cpu()
    head = head_mask.cpu()
    hh = head[src] & head[dst]
    dot_t = ~hh
    return hh.to(edge_index.device), dot_t.to(edge_index.device)


def delta_dpsp(
    model,
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    head_mask: torch.Tensor,
) -> Dict[str, float]:
    """Degree-polarity statistical parity gap between T_hh and T_.t.

    The paper reports the absolute prediction-rate gap for signed link
    prediction groups. We expose both polarity-rate parity and accuracy gap;
    `delta_dpsp` is the average of positive/negative prediction-rate gaps.
    """
    logits, labels = edge_logits(model, z, pos_edge_index, neg_edge_index)
    pred = logits.argmax(dim=1)
    all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    hh, dot_t = _edge_group_masks(all_edges, head_mask)

    if hh.sum() == 0 or dot_t.sum() == 0:
        return {"delta_dpsp": 0.0, "hh_acc": 0.0, "dot_t_acc": 0.0, "acc_gap": 0.0}

    gaps = []
    for sign_class in (0, 1):
        gaps.append((pred[hh].eq(sign_class).float().mean() - pred[dot_t].eq(sign_class).float().mean()).abs())
    delta = torch.stack(gaps).mean()
    hh_acc = pred[hh].eq(labels[hh]).float().mean()
    dot_t_acc = pred[dot_t].eq(labels[dot_t]).float().mean()
    pred_pos_rate_hh = pred[hh].eq(0).float().mean()
    pred_pos_rate_dot_t = pred[dot_t].eq(0).float().mean()
    pred_neg_rate_hh = pred[hh].eq(1).float().mean()
    pred_neg_rate_dot_t = pred[dot_t].eq(1).float().mean()
    return {
        "delta_dpsp": float(delta.item()),
        "hh_acc": float(hh_acc.item()),
        "dot_t_acc": float(dot_t_acc.item()),
        "acc_gap": float((hh_acc - dot_t_acc).abs().item()),
        "pred_pos_rate_hh": float(pred_pos_rate_hh.item()),
        "pred_pos_rate_dot_t": float(pred_pos_rate_dot_t.item()),
        "pred_neg_rate_hh": float(pred_neg_rate_hh.item()),
        "pred_neg_rate_dot_t": float(pred_neg_rate_dot_t.item()),
    }
