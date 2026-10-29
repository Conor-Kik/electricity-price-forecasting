"""Shared preprocessing API for NEM price-forecasting models."""

from .adapters import (
    PreparedModelData,
    extract_actuals_and_p5min,
    prepare_model_data,
)
from .config import FeatureConfig
from .features import (
    add_calendar_features,
    add_core_engineered_features,
    add_forecast_reliability_features,
    add_interconnector_utilisation,
    add_p5min_forecast_features,
    add_targets,
    align_p5min_forecasts,
    build_features,
    canonicalise_columns,
    essential_feature_columns,
    forecast_reliability_columns,
    join_actual_tables,
    validate_actuals,
)
from .scaling import (
    NumericPreprocessor,
    TargetTransformer,
    suggested_scaling_groups,
)
from .sequences import SequenceBatch, make_sequences

__all__ = [
    "FeatureConfig",
    "PreparedModelData",
    "prepare_model_data",
    "extract_actuals_and_p5min",
    "build_features",
    "essential_feature_columns",
    "canonicalise_columns",
    "join_actual_tables",
    "validate_actuals",
    "add_targets",
    "add_core_engineered_features",
    "add_interconnector_utilisation",
    "add_calendar_features",
    "align_p5min_forecasts",
    "add_p5min_forecast_features",
    "add_forecast_reliability_features",
    "forecast_reliability_columns",
    "NumericPreprocessor",
    "TargetTransformer",
    "suggested_scaling_groups",
    "SequenceBatch",
    "make_sequences",
]
