import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


DATA_FILES = {
    "Bitcoinotc": "Bitcoinotc.txt",
    "Bitcoinalpha": "Bitcoinalpha.txt",
    "WikiRfa": "WikiRfa.txt",
    "WikiElec": "WikiElec.txt",
    "Slashdot": "Slashdot.txt",
    "amazon_book": "amazon_book.txt",
}

DATASET_ALIASES = {
    "bitcoin_otc": "Bitcoinotc",
    "bitcoinotc": "Bitcoinotc",
    "bitcoin_alpha": "Bitcoinalpha",
    "bitcoinalpha": "Bitcoinalpha",
    "wiki_rfa": "WikiRfa",
    "wikirfa": "WikiRfa",
    "wiki_elec": "WikiElec",
    "wikielec": "WikiElec",
    "slashdot": "Slashdot",
    "amazon_book": "amazon_book",
    "amazonbook": "amazon_book",
    "amazon-book": "amazon_book",
}


@dataclass
class SignedDataset:
    name: str
    num_nodes: int
    edge_index: torch.Tensor
    edge_label: torch.Tensor
    train_pos_edge_index: torch.Tensor
    train_neg_edge_index: torch.Tensor
    train_pos_pape: torch.Tensor
    train_neg_pape: torch.Tensor
    val_pos_edge_index: torch.Tensor
    val_neg_edge_index: torch.Tensor
    test_pos_edge_index: torch.Tensor
    test_neg_edge_index: torch.Tensor
    pos_degree: torch.Tensor
    neg_degree: torch.Tensor
    degree: torch.Tensor
    head_mask: torch.Tensor
    tail_mask: torch.Tensor
    threshold_k: int


def _read_edges(path: str, dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
    edges, signs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.replace(",", " ").split()
            if len(parts) < 3:
                continue
            src, dst = int(parts[0]), int(parts[1])
            value = float(parts[2])
            if dataset == "amazon_book":
                if value >= 4:
                    sign = 1
                elif value <= 2:
                    sign = -1
                else:
                    continue
            else:
                sign = 1 if value > 0 else -1
            edges.append((src, dst))
            signs.append(sign)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(signs, dtype=torch.long)
    return edge_index, edge_label


def _split_by_sign(edge_index: torch.Tensor, edge_label: torch.Tensor, val_ratio: float, test_ratio: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    pos_edges = edge_index[:, edge_label > 0]
    neg_edges = edge_index[:, edge_label < 0]

    def split(edges: torch.Tensor):
        perm = torch.randperm(edges.size(1), generator=gen)
        test_size = max(1, int(edges.size(1) * test_ratio))
        val_size = max(1, int(edges.size(1) * val_ratio))
        test_idx = perm[:test_size]
        val_idx = perm[test_size:test_size + val_size]
        train_idx = perm[test_size + val_size:]
        return edges[:, train_idx].contiguous(), edges[:, val_idx].contiguous(), edges[:, test_idx].contiguous()

    return (*split(pos_edges), *split(neg_edges))


def _to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    return torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1).contiguous()


def signed_degrees(num_nodes: int, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
    pos_degree = torch.zeros(num_nodes, dtype=torch.long)
    neg_degree = torch.zeros(num_nodes, dtype=torch.long)
    for degree, edges in ((pos_degree, pos_edge_index), (neg_degree, neg_edge_index)):
        src, dst = edges
        degree.scatter_add_(0, src.cpu(), torch.ones_like(src.cpu()))
        degree.scatter_add_(0, dst.cpu(), torch.ones_like(dst.cpu()))
    degree = pos_degree + neg_degree
    return pos_degree, neg_degree, degree


def _signed_adjacency(pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> List[Dict[int, int]]:
    num_nodes = int(torch.cat([pos_edge_index, neg_edge_index], dim=1).max().item()) + 1
    adj: List[Dict[int, int]] = [dict() for _ in range(num_nodes)]
    for edges, sign in ((pos_edge_index, 1), (neg_edge_index, -1)):
        for src, dst in edges.t().tolist():
            adj[src][dst] = sign
    return adj


def build_pape_scores(
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    hops: int = 1,
    max_paths_per_edge: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cached PAPE-style signed path scores for training edges.

    For each attention edge (u, v), the score aggregates signed path
    consistency up to `hops`:
        direct sign / 1 + two-hop signs / 2 + three-hop signs / 3.
    Long path enumeration is capped per edge to keep large graphs practical.
    """
    hops = max(1, min(3, int(hops)))
    adj = _signed_adjacency(pos_edge_index, neg_edge_index)

    def score_edge(src: int, dst: int, direct_sign: int) -> float:
        score = float(direct_sign)
        seen_paths = 0
        if hops >= 2:
            for mid, sign_1 in adj[src].items():
                sign_2 = adj[mid].get(dst)
                if sign_2 is not None:
                    score += (sign_1 * sign_2) / 2.0
                    seen_paths += 1
                    if seen_paths >= max_paths_per_edge:
                        return score
        if hops >= 3:
            for mid_1, sign_1 in adj[src].items():
                for mid_2, sign_2 in adj[mid_1].items():
                    sign_3 = adj[mid_2].get(dst)
                    if sign_3 is not None:
                        score += (sign_1 * sign_2 * sign_3) / 3.0
                        seen_paths += 1
                        if seen_paths >= max_paths_per_edge:
                            return score
        return score

    def score_edges(edges: torch.Tensor, direct_sign: int) -> torch.Tensor:
        values = [score_edge(int(src), int(dst), direct_sign) for src, dst in edges.t().tolist()]
        return torch.tensor(values, dtype=torch.float)

    return score_edges(pos_edge_index, 1), score_edges(neg_edge_index, -1)


def build_head_tail_masks(degree: torch.Tensor, k: int = None, percentile: float = None):
    if percentile is not None:
        k_value = torch.quantile(degree.float(), percentile).item()
        k = int(k_value)
    if k is None:
        nonzero = degree[degree > 0]
        k = int(nonzero.float().mean().round().item()) if nonzero.numel() > 0 else 0
    head_mask = degree > k
    tail_mask = ~head_mask
    return head_mask, tail_mask, k


def load_signed_dataset(
    dataset: str,
    root: str = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    k: int = None,
    percentile: float = None,
    pape_hops: int = 1,
    pape_max_paths: int = 256,
) -> SignedDataset:
    if root is None:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    dataset = DATASET_ALIASES.get(dataset, dataset)
    if dataset not in DATA_FILES:
        raise ValueError(f"Unknown dataset {dataset}. Available: {sorted(DATA_FILES)}")

    path = os.path.join(root, DATA_FILES[dataset])
    edge_index, edge_label = _read_edges(path, dataset)
    num_nodes = int(edge_index.max().item()) + 1

    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = _split_by_sign(edge_index, edge_label, val_ratio, test_ratio, seed)
    train_pos = _to_undirected(train_pos)
    train_neg = _to_undirected(train_neg)
    pos_degree, neg_degree, degree = signed_degrees(num_nodes, train_pos, train_neg)
    train_pos_pape, train_neg_pape = build_pape_scores(
        train_pos,
        train_neg,
        hops=pape_hops,
        max_paths_per_edge=pape_max_paths,
    )
    head_mask, tail_mask, threshold_k = build_head_tail_masks(degree, k=k, percentile=percentile)

    return SignedDataset(
        name=dataset,
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_label=edge_label,
        train_pos_edge_index=train_pos,
        train_neg_edge_index=train_neg,
        train_pos_pape=train_pos_pape,
        train_neg_pape=train_neg_pape,
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
        test_pos_edge_index=test_pos,
        test_neg_edge_index=test_neg,
        pos_degree=pos_degree,
        neg_degree=neg_degree,
        degree=degree,
        head_mask=head_mask,
        tail_mask=tail_mask,
        threshold_k=threshold_k,
    )
