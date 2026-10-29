"""Multi-head attention over a fixed region graph, with no graph-library dependency."""

from __future__ import annotations

import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    """Masked attention between nodes, followed by a residual connection.

    Input and output: (batch, num_nodes, features).

    ``adjacency`` is a (num_nodes, num_nodes) 0/1 matrix with self-loops.
    Non-edges receive -inf before the softmax, so a node attends only to
    itself and its directly connected regions.

    With ``edge_size`` > 0, ``forward`` also takes edge attributes of shape
    (batch, num_nodes, num_nodes, edge_size). They shift each head's score for
    that pair, letting attention depend on the state of the link itself.
    """

    def __init__(
        self,
        features: int,
        adjacency: torch.Tensor,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_size: int = 0,
    ) -> None:
        super().__init__()
        if features % num_heads != 0:
            raise ValueError("features must be divisible by num_heads")
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if not adjacency.diagonal().bool().all():
            raise ValueError("adjacency must include self-loops")

        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.features = features

        self.register_buffer(
            "attention_mask",
            torch.where(
                adjacency.bool(),
                torch.zeros_like(adjacency),
                torch.full_like(adjacency, float("-inf")),
            ),
        )

        self.query_proj = nn.Linear(features, features)
        self.key_proj = nn.Linear(features, features)
        self.value_proj = nn.Linear(features, features)
        self.output_proj = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)
        self.activation = nn.GELU()
        self.edge_proj = nn.Linear(edge_size, num_heads) if edge_size else None

    def forward(
        self, x: torch.Tensor, edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.query_proj(x))  # (batch, heads, nodes, head_dim)
        k = split_heads(self.key_proj(x))
        v = split_heads(self.value_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if self.edge_proj is not None and edge_attr is not None:
            scores = scores + self.edge_proj(edge_attr).permute(0, 3, 1, 2)
        attention = torch.softmax(scores + self.attention_mask, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).reshape(batch_size, num_nodes, self.features)
        return self.norm(x + self.activation(self.output_proj(context)))
