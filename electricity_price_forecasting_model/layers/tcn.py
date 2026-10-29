from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class CausalConv1d(nn.Module):
    """Preserve sequence length using padding only before the first timestep"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int
    ) -> None:
        """Create a convolution whose output cannot depend on future inputs"""
        super().__init__()
        self.left_padding: int = (kernel_size - 1) * dilation
        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolution after padding the past side of the sequence"""
        padded: torch.Tensor = F.pad(inputs, (self.left_padding, 0))
        return self.conv(padded)

class TemporalBlock(nn.Module):
    """Apply two causal convolutions and add a residual connection"""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float
    ) -> None:
        """Create one residual block at the requested dilation"""
        super().__init__()
        self.conv1: CausalConv1d = CausalConv1d(in_channels, channels, kernel_size, dilation)
        self.conv2: CausalConv1d = CausalConv1d(channels, channels, kernel_size, dilation)
        self.activation: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == channels
            else nn.Conv1d(in_channels, channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Combine learned temporal features with the block's input"""
        features: torch.Tensor = self.dropout(self.activation(self.conv1(inputs)))
        features = self.dropout(self.activation(self.conv2(features)))
        return self.activation(features + self.shortcut(inputs))

class TCNNetwork(nn.Module):
    """Produce horizon forecasts from the final timestamp of a feature window."""

    def __init__(
        self,
        input_size: int,
        channels: int = 16,
        kernel_size: int = 3,
        num_blocks: int = 2,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        """Build residual blocks with dilations 1, 2, 4 and so on"""
        super().__init__()
        if input_size <= 0 or channels <= 0 or kernel_size < 2 or num_blocks <= 0:
            raise ValueError("Invalid TCN dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("droput must be in [0,1)")

        self.input_size: int = input_size
        self.receptive_field: int = (1 + 2 * (kernel_size - 1) * (2**num_blocks - 1))

        blocks: list[TemporalBlock] = [
            TemporalBlock(
                input_size if index == 0 else channels,
                channels,
                kernel_size,
                dilation=2**index,
                dropout=dropout
            ) for index in range(num_blocks)
        ]
        self.blocks: nn.Sequential = nn.Sequential(*blocks)
        self.output: nn.Linear = nn.Linear(channels, output_size)

    def temporal_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return features shaped (batch, channels, time)"""
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_size:
            raise ValueError("Expected (batch, time, input_size)")
        # Attribution batches can be views starting at an unaligned storage
        # offset. The residual 1x1 convolution sees that view without padding,
        # so give oneDNN an aligned allocation while preserving autograd.
        # contiguous() alone can return the same unaligned contiguous view.
        if inputs.device.type == "xpu" and inputs.data_ptr() % 64:
            inputs = inputs.clone(memory_format=torch.contiguous_format)
        channels_first: torch.Tensor = inputs.transpose(1,2)
        return self.blocks(channels_first)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one vector of forecasts per input window."""
        latest_features: torch.Tensor = self.temporal_features(inputs)[:, :, -1]
        return self.output(latest_features)
