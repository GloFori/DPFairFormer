from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .layers import DPFairSignedLayer, PolarityAwareEncoder


def negative_sampling(edge_index: torch.Tensor, num_nodes: int, num_neg_samples: Optional[int] = None) -> torch.Tensor:
    num_neg_samples = edge_index.size(1) if num_neg_samples is None else num_neg_samples
    device = edge_index.device
    existing = set((int(s), int(d)) for s, d in edge_index.detach().cpu().t().tolist())
    samples = []
    while len(samples) < num_neg_samples:
        src = torch.randint(0, num_nodes, (num_neg_samples * 2,), device=device)
        dst = torch.randint(0, num_nodes, (num_neg_samples * 2,), device=device)
        for s, d in zip(src.tolist(), dst.tolist()):
            if s != d and (s, d) not in existing:
                samples.append((s, d))
                if len(samples) == num_neg_samples:
                    break
    return torch.tensor(samples, dtype=torch.long, device=device).t().contiguous()


def structured_negative_sampling(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    i, j = edge_index
    k = torch.randint(0, num_nodes, j.shape, device=edge_index.device)
    k = torch.where(k == j, (k + 1) % num_nodes, k)
    return i, j, k


class DPFairSGNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        num_layers: int = 2,
        lamb: float = 5.0,
        transfer_weight: float = 1.0,
        use_transfer: bool = True,
    ):
        super().__init__()
        if hidden_channels % 2 != 0:
            raise ValueError("hidden_channels must be even because SGCN keeps positive and negative streams.")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb
        self.use_transfer = use_transfer

        stream_channels = hidden_channels // 2
        self.layers = nn.ModuleList()
        self.layers.append(DPFairSignedLayer(in_channels, stream_channels, first_layer=True, transfer_weight=transfer_weight))
        for _ in range(num_layers - 1):
            self.layers.append(DPFairSignedLayer(stream_channels, stream_channels, first_layer=False, transfer_weight=transfer_weight))
        self.lin = nn.Linear(2 * hidden_channels, 3)

    def create_spectral_features(self, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor, num_nodes: Optional[int] = None):
        import scipy.sparse as sp
        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).cpu()
        num_nodes = int(edge_index.max().item()) + 1 if num_nodes is None else num_nodes

        # Match the original SGCN spectral feature construction: positive
        # edges are encoded as 2, negative edges as 0, then coalesced and
        # shifted by -1. The training graph is already undirected in
        # dpfair.data, so we build a symmetric unique edge set here
        # instead of blindly appending reverse edges again.
        signed_edges = {}
        for edges, value in ((pos_edge_index.cpu(), 2.0), (neg_edge_index.cpu(), 0.0)):
            for src, dst in edges.t().tolist():
                signed_edges[(int(src), int(dst))] = value
                signed_edges[(int(dst), int(src))] = value

        if not signed_edges:
            return torch.zeros((num_nodes, self.in_channels), dtype=torch.float)

        rows, cols, vals = [], [], []
        for (src, dst), value in signed_edges.items():
            rows.append(src)
            cols.append(dst)
            vals.append(value)

        A = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
        A.sum_duplicates()
        A.data = A.data - 1.0
        dim = min(self.in_channels, max(1, num_nodes - 1))
        svd = TruncatedSVD(n_components=dim, n_iter=128)
        svd.fit(A)
        x = torch.from_numpy(svd.components_.T).float()
        if dim < self.in_channels:
            x = F.pad(x, (0, self.in_channels - dim))
        return x

    def forward(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        tail_mask: Optional[torch.Tensor] = None,
        return_context: bool = False,
        **_: Dict,
    ):
        contexts: List[torch.Tensor] = []
        z = x
        for layer in self.layers:
            z, m = layer(z, pos_edge_index, neg_edge_index, tail_mask=tail_mask, use_transfer=self.use_transfer)
            z = F.relu(z)
            contexts.append(m)
        if return_context:
            return z, contexts
        return z

    def discriminate(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        return F.log_softmax(self.lin(value), dim=1)

    def nll_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))
        loss = F.nll_loss(self.discriminate(z, pos_edge_index), pos_edge_index.new_full((pos_edge_index.size(1),), 0))
        loss = loss + F.nll_loss(self.discriminate(z, neg_edge_index), neg_edge_index.new_full((neg_edge_index.size(1),), 1))
        loss = loss + F.nll_loss(self.discriminate(z, none_edge_index), none_edge_index.new_full((none_edge_index.size(1),), 2))
        return loss / 3.0

    def embedding_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
        pos_loss = torch.clamp((z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1), min=0).mean()
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
        neg_loss = torch.clamp((z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1), min=0).mean()
        return pos_loss + neg_loss

    def task_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        return self.nll_loss(z, pos_edge_index, neg_edge_index) + self.lamb * self.embedding_loss(z, pos_edge_index, neg_edge_index)

    def fairness_loss(self, z: torch.Tensor, edge_index: torch.Tensor, head_mask: torch.Tensor):
        src, dst = edge_index
        head = head_mask.to(z.device)
        cross = head[src] ^ head[dst]
        if cross.sum() == 0:
            return z.new_tensor(0.0)
        cross_src, cross_dst = src[cross], dst[cross]
        head_nodes = torch.cat([cross_src[head[cross_src]], cross_dst[head[cross_dst]]])
        tail_nodes = torch.cat([cross_src[~head[cross_src]], cross_dst[~head[cross_dst]]])
        if head_nodes.numel() == 0 or tail_nodes.numel() == 0:
            return z.new_tensor(0.0)
        return (z[head_nodes].mean(dim=0) - z[tail_nodes].mean(dim=0)).pow(2).sum()

    def reconstruction_loss(self, contexts: List[torch.Tensor], head_mask: torch.Tensor):
        head = head_mask.to(contexts[0].device)
        if head.sum() == 0:
            return contexts[0].new_tensor(0.0)
        return torch.stack([m[head].pow(2).sum(dim=1).mean() for m in contexts]).mean()

    def total_loss(
        self,
        z: torch.Tensor,
        contexts: List[torch.Tensor],
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        head_mask: torch.Tensor,
        mu: float,
        eta: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        task = self.task_loss(z, pos_edge_index, neg_edge_index)
        fair = self.fairness_loss(z, torch.cat([pos_edge_index, neg_edge_index], dim=1), head_mask)
        recon = self.reconstruction_loss(contexts, head_mask)
        total = task + mu * fair + eta * recon
        parts = {"task": float(task.detach()), "fair": float(fair.detach()), "recon": float(recon.detach())}
        return total, parts


class DPFairFormer(DPFairSGNN):
    def __init__(self, *args, encoder_layers: int = 1, attention_dropout: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = PolarityAwareEncoder(
            self.in_channels,
            num_layers=encoder_layers,
            dropout=attention_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        tail_mask: Optional[torch.Tensor] = None,
        return_context: bool = False,
        pos_degree: Optional[torch.Tensor] = None,
        neg_degree: Optional[torch.Tensor] = None,
        pos_pape: Optional[torch.Tensor] = None,
        neg_pape: Optional[torch.Tensor] = None,
    ):
        if pos_degree is not None and neg_degree is not None:
            x = self.encoder(
                x,
                pos_degree,
                neg_degree,
                pos_edge_index,
                neg_edge_index,
                pos_pape=pos_pape,
                neg_pape=neg_pape,
            )
        return super().forward(
            x,
            pos_edge_index,
            neg_edge_index,
            tail_mask=tail_mask,
            return_context=return_context,
        )
