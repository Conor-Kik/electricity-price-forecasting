"""PyTorch GRU layer used by the corresponding forecasting wrapper."""

from __future__ import annotations

import torch
from torch import nn


class GRUNetwork(nn.Module):
    """GRU followed by a linear head over the final sequence timestep."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.gru(inputs)
        return self.output(sequence_output[:, -1, :])
