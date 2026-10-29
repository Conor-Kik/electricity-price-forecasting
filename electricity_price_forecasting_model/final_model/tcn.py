"""TCN forecaster using the shared PyTorch training pipline"""
from __future__ import annotations

from torch import nn
from electricity_price_forecasting_model.layers.tcn import TCNNetwork
from pipeline.neural import TorchForecastModel

class TCNModel(TorchForecastModel):
    """Fit and predict with a TCN through the shared sequence-model"""

    model_name: str = "TCN"
    requires_sequence: bool= True

    def __init__(
        self,
        sequence_length: int = 12,
        channels: int = 16,
        kernel_size: int = 3,
        num_blocks: int = 2,
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
        show_progress: bool = True
    ) -> None:
        """Validate TCN settings and configure training options"""
        architecture_sizes: tuple[tuple[str, int], ...] = (
            ("sequence_length", sequence_length),
            ("channels", channels),
            ("kernel_size", kernel_size),
            ("num_blocks", num_blocks)
        )

        for name, value in architecture_sizes:
            if value <= 0 or int(value) != value:
                raise ValueError(f"{name} must be a positive integer")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.sequence_length: int = int(sequence_length)
        self.channels: int = int(channels)
        self.kernel_size: int = int(kernel_size)
        self.num_blocks: int = int(num_blocks)
        self.dropout: float = float(dropout)
        self.show_progress: bool = show_progress

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
            interval_minutes=interval_minutes
        )

    def _build_network(self, input_size: int) -> nn.Module:
        """Create an untrained TCN for the prepared feature count"""
        return TCNNetwork(
            input_size=input_size,
            channels=self.channels,
            kernel_size=self.kernel_size,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            output_size=self.output_size,
        )
