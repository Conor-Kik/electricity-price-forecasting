"""Trainable MLP model assembled from a PyTorch layer and the shared pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from electricity_price_forecasting_model.layers.mlp import FeedForwardNetwork
from pipeline.neural import TorchForecastModel


class MLPModel(TorchForecastModel):
    model_name = "MLP"

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (128, 64, 32),
        dropout: float | Sequence[float] = (0.2, 0.15, 0.0),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        prediction_batch_size: int = 1024,
        max_epochs: int = 30,
        patience: int = 3,
        min_delta: float = 0.001,
        loss: str = "huber",
        huber_delta: float = 1.0,
        random_seed: int = 42,
        device: str | None = None,
        selection_metric: str = "mae",
        gradient_clip_norm: float | None = None,
        scheduler_step_size: int | None = None,
        scheduler_gamma: float = 0.5,
        mixed_precision: bool = False,
        interval_minutes: int = 5,
    ) -> None:
        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        rates = (
            (float(dropout),) * len(hidden_sizes)
            if isinstance(dropout, (int, float))
            else tuple(float(rate) for rate in dropout)
        )
        if len(rates) != len(hidden_sizes):
            raise ValueError("dropout must provide one rate per hidden layer")
        if any(not 0 <= rate < 1 for rate in rates):
            raise ValueError("dropout rates must be in [0, 1)")
        self.hidden_sizes = hidden_sizes
        self.dropout = rates
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            prediction_batch_size=prediction_batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            loss=loss,
            huber_delta=huber_delta,
            random_seed=random_seed,
            device=device,
            selection_metric=selection_metric,
            gradient_clip_norm=gradient_clip_norm,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
            mixed_precision=mixed_precision,
            interval_minutes=interval_minutes,
        )

    def _build_network(self, input_size: int) -> nn.Module:
        return FeedForwardNetwork(input_size, self.hidden_sizes, self.dropout, output_size=self.output_size)
