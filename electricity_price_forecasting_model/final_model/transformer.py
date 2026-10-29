from __future__ import annotations

from torch import nn

from pipeline.neural import TorchForecastModel
from electricity_price_forecasting_model.layers.transformer_layers.forecaster import Forecaster


class TransformerModel(TorchForecastModel):
    model_name = "Transformer"
    requires_sequence = True

    def __init__(
        self,
        sequence_length: int = 36,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        prediction_batch_size: int = 256,
        max_epochs: int = 20,
        patience: int = 3,
        min_delta: float = 0.001,
        loss: str = "huber",
        huber_delta: float = 1.0,
        random_seed: int = 42,
        device: str | None = None,
        selection_metric: str = "mae",
        gradient_clip_norm: float | None = 1.0,
        scheduler_step_size: int | None = None,
        scheduler_gamma: float = 0.5,
        mixed_precision: bool = False,
        interval_minutes: int = 5,
    ) -> None:
        if sequence_length <= 0 or int(sequence_length) != sequence_length:
            raise ValueError("sequence_length must be a positive integer")
        if d_model <= 0 or nhead <= 0 or num_layers <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
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
        return Forecaster(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_size=self.output_size,
        )
