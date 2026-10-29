"""Reusable lazy sequence datasets for PyTorch forecasting models.

This module owns the temporal alignment shared by recurrent, convolutional,
and attention-based sequence models.  Network architecture, batching, and
training policy remain the responsibility of each model wrapper.

These tensor-only primitives assume an already validated, single-series
timeline. Production models use pipeline.neural, which validates timestamp and
region metadata before converting frames to tensors.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Expose fixed-length training windows without materialising them.

    The target at a window's final timestep is returned.  ``target_mask`` can
    exclude invalid targets while retaining their feature rows as context for
    later valid targets.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sequence_length: int,
        target_mask: np.ndarray | None = None,
    ) -> None:
        if sequence_length <= 0 or int(sequence_length) != sequence_length:
            raise ValueError("sequence_length must be a positive integer")
        sequence_length = int(sequence_length)
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same number of rows")
        if len(inputs) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} training rows, got {len(inputs)}"
            )

        candidate_indices = np.arange(sequence_length - 1, len(inputs))
        if target_mask is not None:
            mask = np.asarray(target_mask, dtype=bool)
            if mask.ndim != 1 or len(mask) != len(inputs):
                raise ValueError("target_mask must have one value per input row")
            candidate_indices = candidate_indices[mask[candidate_indices]]
        if len(candidate_indices) == 0:
            raise ValueError("No valid targets have enough preceding sequence context")

        self.inputs = inputs
        self.targets = targets
        self.sequence_length = int(sequence_length)
        self.target_indices = candidate_indices

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_index = int(self.target_indices[index])
        start = target_index - self.sequence_length + 1
        return self.inputs[start : target_index + 1], self.targets[target_index]


class ContextSequenceDataset(Dataset):
    """Expose one prediction window per input row using preceding history.

    Supplying the final ``sequence_length - 1`` training rows as ``history``
    keeps the first validation/test prediction aligned instead of discarding
    rows at every chronological split.
    """

    def __init__(
        self,
        history: torch.Tensor,
        inputs: torch.Tensor,
        sequence_length: int,
    ) -> None:
        if sequence_length <= 0 or int(sequence_length) != sequence_length:
            raise ValueError("sequence_length must be a positive integer")
        sequence_length = int(sequence_length)
        if history.ndim != inputs.ndim or history.shape[1:] != inputs.shape[1:]:
            raise ValueError("history and inputs must have matching feature shapes")

        context_length = sequence_length - 1
        if len(history) < context_length:
            raise ValueError(
                f"Need at least {context_length} history rows, got {len(history)}"
            )
        context = history[-context_length:] if context_length else history[:0]
        self.combined = torch.cat((context, inputs), dim=0)
        self.input_count = len(inputs)
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return self.input_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        return (self.combined[index : index + self.sequence_length],)
