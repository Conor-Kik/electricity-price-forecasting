"""Trainable GRU model assembled from a PyTorch layer and the shared pipeline."""

from __future__ import annotations

from torch import nn

from electricity_price_forecasting_model.layers.gru import GRUNetwork
from pipeline.neural import TorchForecastModel


class GRUModel(TorchForecastModel):
    model_name = "GRU"
    requires_sequence = True

    def __init__(
        self,
        sequence_length: int = 36,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        prediction_batch_size: int = 1024,
        max_epochs: int = 20,
        patience: int = 4,
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
        if (
            sequence_length <= 0
            or int(sequence_length) != sequence_length
            or hidden_size <= 0
            or num_layers <= 0
        ):
            raise ValueError(
                "sequence_length, hidden_size, and num_layers must be positive integers"
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
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
        return GRUNetwork(
            input_size, self.hidden_size, self.num_layers, output_size=self.output_size, dropout=self.dropout
        )
