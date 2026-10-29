"""Reusable training utilities for deep-learning forecasting models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from torch import nn


@dataclass(frozen=True)
class TrainingResult:
    """Training history and the settings needed to refit the selected model.

    Epoch numbers are one-based. ``best_epoch`` can precede ``stopped_epoch``:
    patience allows further training before the best weights are restored.
    """

    best_epoch: int
    stopped_epoch: int
    best_validation_score: float | None
    training_loss_history: list[float]
    validation_score_history: list[float]
    refit_config: dict[str, object] = field(default_factory=dict)


class TorchEarlyStoppingTrainer:
    """Run a PyTorch epoch loop with optional validation checkpointing.

    Models retain ownership of their architecture, data loaders, optimizer,
    loss, and validation metric. The callbacks make this trainer reusable for
    feed-forward, recurrent, convolutional, and Transformer architectures.
    Validation scores are assumed to be minimised.
    """

    def __init__(
        self,
        *,
        max_epochs: int,
        patience: int = 3,
        min_delta: float = 0.001,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

    def fit(
        self,
        *,
        network: nn.Module,
        train_epoch: Callable[[], float],
        validation_score: Callable[[], float] | None = None,
    ) -> TrainingResult:
        """Train, restore the best checkpoint, and return run metadata."""
        training_history: list[float] = []
        validation_history: list[float] = []
        best_epoch = 0
        best_validation_score = float("inf")
        best_state: dict[str, object] | None = None
        epochs_without_improvement = 0
        stopped_epoch = self.max_epochs

        for epoch in range(1, self.max_epochs + 1):
            training_loss = float(train_epoch())
            if not np.isfinite(training_loss):
                raise RuntimeError(
                    f"Training produced a non-finite loss at epoch {epoch}"
                )
            training_history.append(training_loss)

            if validation_score is None:
                # Final refits use the epoch budget chosen during model selection.
                # They must not use test outcomes to decide when to stop.
                best_epoch = epoch
                continue

            score = float(validation_score())
            if not np.isfinite(score):
                raise RuntimeError(
                    f"Validation produced a non-finite score at epoch {epoch}"
                )
            validation_history.append(score)

            if score < best_validation_score - self.min_delta:
                best_validation_score = score
                best_epoch = epoch
                # Clone the checkpoint so later updates cannot change it. Keeping
                # it on CPU avoids reserving a second model's worth of GPU memory.
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    stopped_epoch = epoch
                    break

        if best_state is not None:
            network.load_state_dict(best_state)

        return TrainingResult(
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            best_validation_score=(
                best_validation_score if validation_score is not None else None
            ),
            training_loss_history=training_history,
            validation_score_history=validation_history,
            refit_config={"max_epochs": best_epoch},
        )
