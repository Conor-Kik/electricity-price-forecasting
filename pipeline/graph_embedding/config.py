from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .graph import DEFAULT_REGIONS

# Compact per-node inputs. The downstream model still receives the complete
# engineered feature set directly; the encoder only has to summarise the
# cross-region state. Names absent from the prepared frame are skipped.
DEFAULT_NODE_FEATURES: tuple[str, ...] = (
    "rrp",
    "totaldemand",
    "availablegeneration",
    "netinterchange",
    "reserve_margin_mw",
    "reserve_ratio",
    "rrp_delta_1",
    "rrp_delta_3",
    "rrp_delta_6",
    "demand_delta_1",
    "demand_delta_3",
    "rrp_roll_mean_12",
    "rrp_roll_std_12",
    "rrp_roll_mean_288",
    "rrp_lag_288",
    "time_sin",
    "time_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
)


@dataclass(frozen=True)
class GraphEmbeddingConfig:
    """Settings for the graph encoder that adds ``gnn_emb_*`` feature columns.

    ``fit_fraction`` is the leading share of the training period the encoder
    may learn from. It must end before the earliest model-selection validation
    block, otherwise downstream validation scores would be optimistic.

    ``edge_features`` lets attention see interconnector flow, utilisation and
    headroom. ``spread_loss_weight`` also trains the encoder to forecast the
    price difference across each interconnector; zero disables it.
    """

    regions: tuple[str, ...] = DEFAULT_REGIONS
    adjacency_mode: Literal["nem"] = "nem"
    edge_features: bool = True
    spread_loss_weight: float = 0.5
    node_features: tuple[str, ...] | None = None
    sequence_length: int = 12
    embedding_dim: int = 8
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    fit_fraction: float = 0.6
    inner_validation_fraction: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    prediction_batch_size: int = 4096
    max_epochs: int = 20
    patience: int = 3
    min_delta: float = 0.001
    huber_delta: float = 1.0
    random_seed: int = 42
    device: str | None = None

    def __post_init__(self) -> None:
        if len(self.regions) < 2 or len(set(self.regions)) != len(self.regions):
            raise ValueError("regions must name at least two distinct nodes")
        if self.adjacency_mode != "nem":
            raise ValueError("The encoder requires the NEM interconnector graph")
        sizes = (
            self.sequence_length,
            self.embedding_dim,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.batch_size,
            self.prediction_batch_size,
            self.max_epochs,
            self.patience,
        )
        if any(value <= 0 or int(value) != value for value in sizes):
            raise ValueError("Encoder sizes and training counts must be positive integers")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.fit_fraction < 1:
            raise ValueError("fit_fraction must be between 0 and 1")
        if not 0 < self.inner_validation_fraction < 1:
            raise ValueError("inner_validation_fraction must be between 0 and 1")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("Invalid learning_rate, weight_decay or min_delta")
        if self.spread_loss_weight < 0:
            raise ValueError("spread_loss_weight must be non-negative")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")

    @property
    def embedding_columns(self) -> list[str]:
        return [f"gnn_emb_{index:02d}" for index in range(self.embedding_dim)]
