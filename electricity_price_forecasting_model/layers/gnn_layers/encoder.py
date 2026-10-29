"""Spatio-temporal graph encoder that produces one embedding per region."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from electricity_price_forecasting_model.layers.gnn_layers.graph_attention import (
    GraphAttentionLayer,
)


class SpatioTemporalGraphEncoder(nn.Module):
    """Graph attention at each timestep, then a GRU over the lookback per node.

    Input: (batch, lookback, num_nodes, input_size + num_nodes * edge_size).
    The trailing values of each node are its edge attributes towards every
    node, packed there so one tensor can be windowed over time.

    Returns ``(embedding, prediction, spread)`` with shapes
    (batch, num_nodes, embedding_dim), (batch, num_nodes) and
    (batch, len(spread_pairs)). The two heads exist only to train the encoder;
    callers keep the embedding. ``spread`` forecasts a quantity between the
    two nodes of each pair, such as their future price difference.
    """

    def __init__(
        self,
        input_size: int,
        adjacency: torch.Tensor,
        hidden_dim: int = 32,
        embedding_dim: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_size: int = 0,
        spread_pairs: Sequence[tuple[int, int]] = (),
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        num_nodes = adjacency.shape[0]
        self.input_size = input_size
        self.edge_size = edge_size
        self.input_proj = nn.Linear(input_size, hidden_dim)
        # Weights are shared across regions; this lets a node still be told apart.
        self.node_embedding = nn.Parameter(torch.zeros(num_nodes, hidden_dim))
        self.graph_layers = nn.ModuleList(
            GraphAttentionLayer(
                hidden_dim,
                adjacency,
                num_heads=num_heads,
                dropout=dropout,
                edge_size=edge_size,
            )
            for _ in range(num_layers)
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(hidden_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)
        pairs = torch.tensor(list(spread_pairs), dtype=torch.long).reshape(-1, 2)
        self.register_buffer("spread_pairs", pairs)
        self.spread_head = nn.Linear(2 * embedding_dim, 1) if len(pairs) else None

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, lookback, num_nodes, _ = x.shape
        edge_attr = None
        if self.edge_size:
            edge_attr = x[..., self.input_size :].reshape(
                batch_size * lookback, num_nodes, num_nodes, self.edge_size
            )
        hidden = self.input_proj(x[..., : self.input_size]) + self.node_embedding
        hidden = hidden.reshape(batch_size * lookback, num_nodes, -1)
        for layer in self.graph_layers:
            hidden = layer(hidden, edge_attr)

        # (batch, lookback, nodes, hidden) -> one sequence per (batch, node).
        hidden = hidden.view(batch_size, lookback, num_nodes, -1).transpose(1, 2)
        hidden = hidden.reshape(batch_size * num_nodes, lookback, -1)
        _, last = self.gru(hidden)

        embedding = self.bottleneck(self.dropout(last[-1]))
        prediction = self.head(embedding).view(batch_size, num_nodes)
        embedding = embedding.view(batch_size, num_nodes, -1)
        if self.spread_head is None:
            spread = prediction.new_zeros(batch_size, 0)
        else:
            ends = torch.cat(
                (
                    embedding[:, self.spread_pairs[:, 0]],
                    embedding[:, self.spread_pairs[:, 1]],
                ),
                dim=-1,
            )
            spread = self.spread_head(ends).squeeze(-1)
        return embedding, prediction, spread
