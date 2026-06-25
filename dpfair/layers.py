import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
    if num_nodes is None:
        num_nodes = x.size(0)
    src, dst = edge_index
    out = x.new_zeros((num_nodes, x.size(1)))
    deg = x.new_zeros((num_nodes, 1))
    out.index_add_(0, dst, x[src])
    deg.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=x.dtype, device=x.device))
    return out / deg.clamp_min(1.0)


class PolarityTransfer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma_self = nn.Linear(channels, channels, bias=False)
        self.gamma_neigh = nn.Linear(channels, channels, bias=False)
        self.beta_self = nn.Linear(channels, channels, bias=False)
        self.beta_neigh = nn.Linear(channels, channels, bias=False)
        self.r = nn.Parameter(torch.empty(1, channels))
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.gamma_self, self.gamma_neigh, self.beta_self, self.beta_neigh):
            module.reset_parameters()
        nn.init.xavier_uniform_(self.r)

    def forward(self, h_self: torch.Tensor, h_neigh: torch.Tensor) -> torch.Tensor:
        gamma = self.act(self.gamma_self(h_self) + self.gamma_neigh(h_neigh)) + 1.0
        beta = self.act(self.beta_self(h_self) + self.beta_neigh(h_neigh))
        r_v = gamma * self.r + beta
        return h_self + r_v - h_neigh


class DPFairSignedLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first_layer: bool, transfer_weight: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layer = first_layer
        self.transfer_weight = transfer_weight

        left_in = in_channels if first_layer else 2 * in_channels
        self.lin_pos_l = nn.Linear(left_in, out_channels, bias=False)
        self.lin_neg_l = nn.Linear(left_in, out_channels, bias=False)
        self.lin_pos_r = nn.Linear(in_channels, out_channels)
        self.lin_neg_r = nn.Linear(in_channels, out_channels)
        self.transfer_pos = PolarityTransfer(in_channels)
        self.transfer_neg = PolarityTransfer(in_channels)

    def _tail_complement(self, m: torch.Tensor, tail_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if tail_mask is None:
            return m
        return m * tail_mask.to(m.device).float().view(-1, 1)

    def forward(
        self,
        x: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        tail_mask: Optional[torch.Tensor] = None,
        use_transfer: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.first_layer:
            x_pos = x_neg = x
            pos_neigh = mean_aggregate(x, pos_edge_index)
            neg_neigh = mean_aggregate(x, neg_edge_index)
            m_pos = self.transfer_pos(x, pos_neigh)
            m_neg = self.transfer_neg(x, neg_neigh)
            if use_transfer:
                pos_neigh = pos_neigh + self.transfer_weight * self._tail_complement(m_pos, tail_mask)
                neg_neigh = neg_neigh + self.transfer_weight * self._tail_complement(m_neg, tail_mask)
            out_pos = self.lin_pos_l(pos_neigh) + self.lin_pos_r(x_pos)
            out_neg = self.lin_neg_l(neg_neigh) + self.lin_neg_r(x_neg)
            return torch.cat([out_pos, out_neg], dim=-1), torch.cat([m_pos, m_neg], dim=-1)

        x_pos, x_neg = x.split(self.in_channels, dim=-1)
        pos_from_pos = mean_aggregate(x_pos, pos_edge_index)
        pos_from_neg = mean_aggregate(x_neg, neg_edge_index)
        neg_from_neg = mean_aggregate(x_neg, pos_edge_index)
        neg_from_pos = mean_aggregate(x_pos, neg_edge_index)

        pos_neigh = torch.cat([pos_from_pos, pos_from_neg], dim=-1)
        neg_neigh = torch.cat([neg_from_neg, neg_from_pos], dim=-1)
        pos_context = pos_neigh[:, : self.in_channels]
        neg_context = neg_neigh[:, : self.in_channels]
        m_pos = self.transfer_pos(x_pos, pos_context)
        m_neg = self.transfer_neg(x_neg, neg_context)
        if use_transfer:
            comp_pos = self._tail_complement(m_pos, tail_mask)
            comp_neg = self._tail_complement(m_neg, tail_mask)
            pos_neigh = pos_neigh + self.transfer_weight * torch.cat([comp_pos, comp_pos], dim=-1)
            neg_neigh = neg_neigh + self.transfer_weight * torch.cat([comp_neg, comp_neg], dim=-1)

        out_pos = self.lin_pos_l(pos_neigh) + self.lin_pos_r(x_pos)
        out_neg = self.lin_neg_l(neg_neigh) + self.lin_neg_r(x_neg)
        return torch.cat([out_pos, out_neg], dim=-1), torch.cat([m_pos, m_neg], dim=-1)


def _scatter_softmax(score: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    max_score = score.new_full((dim_size,), -torch.inf)
    max_score.scatter_reduce_(0, index, score, reduce="amax", include_self=True)
    exp_score = torch.exp(score - max_score[index])
    denom = score.new_zeros((dim_size,))
    denom.scatter_add_(0, index, exp_score)
    return exp_score / denom[index].clamp_min(1e-12)


class PolarityAwareAttentionLayer(nn.Module):
    """Sparse polarity-aware Transformer layer over signed neighborhoods.

    The paper injects PAPE/PADE into attention as a structural bias. This
    implementation keeps the computation sparse: attention is only computed
    on positive edges, negative edges, and self loops.
    """

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.bias_mlp = nn.Sequential(
            nn.Linear(6, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * channels, channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_degree: torch.Tensor,
        neg_degree: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        pos_pape: Optional[torch.Tensor] = None,
        neg_pape: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        total = (pos_degree + neg_degree).float().clamp_min(1).to(x.device)
        degree_feat = torch.stack(
            [pos_degree.float().to(x.device) / total, neg_degree.float().to(x.device) / total],
            dim=-1,
        )

        self_loop = torch.arange(num_nodes, device=x.device)
        self_edge_index = torch.stack([self_loop, self_loop], dim=0)
        edge_index = torch.cat([pos_edge_index, neg_edge_index, self_edge_index], dim=1)
        edge_sign = torch.cat([
            torch.ones(pos_edge_index.size(1), device=x.device),
            -torch.ones(neg_edge_index.size(1), device=x.device),
            torch.zeros(num_nodes, device=x.device),
        ])
        if pos_pape is None:
            pos_pape = torch.ones(pos_edge_index.size(1), device=x.device)
        else:
            pos_pape = pos_pape.to(x.device).float()
        if neg_pape is None:
            neg_pape = -torch.ones(neg_edge_index.size(1), device=x.device)
        else:
            neg_pape = neg_pape.to(x.device).float()
        edge_pape = torch.cat([pos_pape, neg_pape, torch.zeros(num_nodes, device=x.device)])
        src, dst = edge_index

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        edge_pape = edge_pape / edge_pape.abs().max().clamp_min(1.0)
        pair_feat = torch.cat([edge_sign.view(-1, 1), edge_pape.view(-1, 1), degree_feat[src], degree_feat[dst]], dim=1)
        structural_bias = self.bias_mlp(pair_feat).squeeze(-1)
        score = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(x.size(1)) + structural_bias
        alpha = _scatter_softmax(score, dst, num_nodes)

        out = x.new_zeros(x.shape)
        out.index_add_(0, dst, alpha.view(-1, 1) * v[src])
        x = self.norm1(x + self.dropout(self.out_proj(out)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class PolarityAwareEncoder(nn.Module):
    """PADE/PAPE-aware sparse Transformer encoder."""

    def __init__(self, channels: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pade = nn.Sequential(nn.Linear(2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.layers = nn.ModuleList([
            PolarityAwareAttentionLayer(channels, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        pos_degree: torch.Tensor,
        neg_degree: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        pos_pape: Optional[torch.Tensor] = None,
        neg_pape: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total = (pos_degree + neg_degree).float().clamp_min(1).to(x.device)
        degree_feat = torch.stack(
            [pos_degree.float().to(x.device) / total, neg_degree.float().to(x.device) / total],
            dim=-1,
        )
        x = self.norm(x + self.pade(degree_feat))
        for layer in self.layers:
            x = layer(x, pos_degree, neg_degree, pos_edge_index, neg_edge_index, pos_pape=pos_pape, neg_pape=neg_pape)
        return x
