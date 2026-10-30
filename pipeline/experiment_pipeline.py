"""Reusable data preparation and evaluation for dispatch-price models.

Model scripts should define only their estimator, fitting procedure, and any
model-specific diagnostics. This module owns experiment configuration, prepared
data caching, chronological validation, target reconstruction, and common
forecast evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from pipeline.contracts import forecast_array, target_columns
from pipeline.graph_embedding import GraphEmbeddingConfig
from pipeline.preprocessing import (
    PreparedModelData,
    forecast_reliability_columns,
    prepare_model_data,
    add_targets,
)
from pipeline.preprocessing.adapters import LearnedFeatures
from pipeline.preprocessing.config import FeatureConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TargetType = Literal["change", "price", "aemo_error"]
FeatureSelection = Literal["all"] | Sequence[str]
CACHE_SCHEMA_VERSION = 5
HOUR_HORIZONS = tuple(range(5, 61, 5))


def source_fingerprint() -> str:
    """Fingerprint preparation code and local raw-cache file identities.

    Raw files use path, size and nanosecond mtime (not a full content hash).
    Upstream revisions not downloaded locally still require force_refresh.
    """
    digest = hashlib.sha256()
    sources = sorted((PROJECT_ROOT / "pipeline" / "preprocessing").glob("*.py"))
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    ingestion = PROJECT_ROOT / "pipeline" / "data_cleaning" / "data.py"
    digest.update(ingestion.read_bytes())
    for name in ("nemosis_cache", "nemseer_cache"):
        for path in sorted((ingestion.parent / name).rglob("*")):
            if path.is_file():
                stat = path.stat()
                digest.update(
                    f"{path.relative_to(PROJECT_ROOT)}:{stat.st_size}:{stat.st_mtime_ns}".encode()
                )
    return digest.hexdigest()


def graph_source_fingerprint() -> str:
    """Fingerprint the graph-encoder code when graph features are enabled."""
    digest = hashlib.sha256()
    layers = PROJECT_ROOT / "electricity_price_forecasting_model" / "layers" / "gnn_layers"
    for directory in (PROJECT_ROOT / "pipeline" / "graph_embedding", layers):
        for source in sorted(directory.glob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by all models in a comparable experiment."""

    # Inclusive start and exclusive end, in timezone-naive NEM market time.
    start_date: str = "2022-01-01"
    end_date: str = "2026-07-31"
    region: str = "NSW1"
    target_type: TargetType = "change"
    model_features: FeatureSelection = "all"
    validation_fraction: float = 0.15
    cache_directory: Path = PROJECT_ROOT / "model_cache"
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    walk_forward_folds: int = 1
    selection_metric: str = "mae"
    random_seed: int = 42
    # Reliability features are appended after loading the shared prepared cache.
    forecast_reliability: bool = False
    # None disables learned graph features; the default uses the full NEM graph.
    graph_embedding: GraphEmbeddingConfig | None = field(default_factory=GraphEmbeddingConfig)

    def __post_init__(self) -> None:
        try:
            start = datetime.fromisoformat(self.start_date)
            end = datetime.fromisoformat(self.end_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("start_date and end_date must be ISO date/time strings") from exc
        if start.tzinfo is not None or end.tzinfo is not None:
            raise ValueError("Experiment dates must use timezone-naive NEM market time")
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")
        object.__setattr__(self, "start_date", start.isoformat())
        object.__setattr__(self, "end_date", end.isoformat())
        if self.feature_config.interval_minutes != 5:
            raise ValueError("The 12-output experiment requires five-minute data")
        if self.graph_embedding is not None and not isinstance(self.graph_embedding, GraphEmbeddingConfig):
            raise ValueError("graph_embedding must be GraphEmbeddingConfig or None")
        if self.graph_embedding is not None and self.graph_embedding.adjacency_mode != "nem":
            raise ValueError("Every experiment requires the NEM interconnector graph")
        if self.target_type not in {"change", "price", "aemo_error"}:
            raise ValueError("target_type must be 'change', 'price', or 'aemo_error'")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if (
            self.walk_forward_folds < 1
            or int(self.walk_forward_folds) != self.walk_forward_folds
        ):
            raise ValueError("walk_forward_folds must be a positive integer")
        if self.selection_metric not in {"mae", "rmse", "sse", "median_ae"}:
            raise ValueError("Unsupported selection_metric")

    @property
    def horizon_minutes(self) -> int:
        """Longest forecast lead, also used for encoder labels and embargoes."""
        return HOUR_HORIZONS[-1]

    @property
    def forecast_horizons(self) -> tuple[int, ...]:
        """Every experiment predicts the complete hour at five-minute intervals."""
        return HOUR_HORIZONS

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.forecast_horizons

    @property
    def aemo_forecast_column(self) -> str:
        return f"p5_h{self.horizon_minutes:02d}_rrp"

    @property
    def predicted_target_column(self) -> str:
        if self.target_type == "change":
            return "predicted_change"
        if self.target_type == "aemo_error":
            return "predicted_aemo_error"
        return "predicted_price"

    @property
    def feature_cache_key(self) -> str:
        if self.model_features == "all":
            return "all"
        serialised = json.dumps(list(self.model_features), separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:10]

    @property
    def graph_embedding_cache_key(self) -> str:
        """Identify the graph encoder's configuration, or the plain-feature path."""
        if self.graph_embedding is None:
            return "none"
        serialised = json.dumps(asdict(self.graph_embedding), sort_keys=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:10]

    @property
    def date_range_cache_key(self) -> str:
        """Keep prepared data and graph embeddings separate for each date range."""
        value = f"{self.start_date}/{self.end_date}"
        return hashlib.sha256(value.encode()).hexdigest()[:12]

    @property
    def cache_file(self) -> Path:
        embedding = self.graph_embedding_cache_key
        filename = (
            f"prepared_{self.region}_{self.horizon_minutes}min_"
            f"{self.target_type}_{self.feature_cache_key}_{self.date_range_cache_key}_"
            f"{f'gnn{embedding}' if self.graph_embedding is not None else 'plain'}.joblib"
        )
        return self.cache_directory / filename

    @property
    def graph_embedding_cache_file(self) -> Path:
        """Encoder output is independent of ``region`` and shared across them."""
        filename = (
            f"graph_embedding_{self.horizon_minutes}min_{self.target_type}_"
            f"{self.graph_embedding_cache_key}_{self.date_range_cache_key}.joblib"
        )
        return self.cache_directory / filename

    def cache_metadata(self) -> dict[str, object]:
        features: str | list[str]
        if self.model_features == "all":
            features = "all"
        else:
            features = list(self.model_features)
        metadata: dict[str, object] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "region": self.region,
            "horizon_minutes": self.horizon_minutes,
            "target_type": self.target_type,
            "model_features": features,
            "feature_config": asdict(self.feature_config),
            "source_fingerprint": source_fingerprint(),
        }
        metadata["forecast_horizons"] = list(self.forecast_horizons)
        metadata["graph_embedding"] = asdict(self.graph_embedding) if self.graph_embedding is not None else None
        if self.graph_embedding is not None:
            metadata["graph_source_fingerprint"] = graph_source_fingerprint()
        return metadata

    def graph_embedding_cache_metadata(self) -> dict[str, object]:
        metadata = self.cache_metadata()
        del metadata["region"], metadata["model_features"]
        return metadata


@dataclass(frozen=True)
class ExperimentData:
    """Prepared data and the leakage-safe model-selection split."""

    config: ExperimentConfig
    prepared: PreparedModelData
    fit_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame

    @property
    def features(self) -> list[str]:
        return self.prepared.features


def _cache_payload_is_valid(
    payload: object,
    config: ExperimentConfig,
) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("metadata") == config.cache_metadata()
        and isinstance(payload.get("data"), PreparedModelData)
    )


def graph_embedding_stage(config: ExperimentConfig, *, force_refresh: bool = False):
    """Return the learned-feature stage, or None when disabled by the config.

    The encoder is fitted once per horizon/target and cached, so experiments
    on different regions reuse the same embeddings.
    """
    if config.graph_embedding is None:
        return None

    def stage(train_features: pd.DataFrame, test_features: pd.DataFrame):
        path = config.graph_embedding_cache_file
        metadata = config.graph_embedding_cache_metadata()
        if path.exists() and not force_refresh:
            payload = joblib.load(path)
            if (
                isinstance(payload, dict)
                and payload.get("metadata") == metadata
                and isinstance(payload.get("data"), LearnedFeatures)
            ):
                print(f"Loading graph embeddings from {path.name}...")
                return payload["data"]
        print("Fitting graph encoder...")
        # Cached prepared features do not need to load the encoder runtime.
        from pipeline.graph_embedding.embedder import compute_graph_embeddings

        learned = compute_graph_embeddings(
            train_features,
            test_features,
            config=config.graph_embedding,
            horizon=config.horizon_minutes,
            target=config.target_type,
            interval_minutes=config.feature_config.interval_minutes,
        )
        joblib.dump({"metadata": metadata, "data": learned}, path)
        return learned

    return stage


def load_prepared_data(
    config: ExperimentConfig,
    *,
    force_refresh: bool = False,
) -> PreparedModelData:
    """Load a compatible prepared-data cache or construct and store one."""
    config.cache_directory.mkdir(parents=True, exist_ok=True)

    if config.cache_file.exists() and not force_refresh:
        payload = joblib.load(config.cache_file)
        if _cache_payload_is_valid(payload, config):
            print(f"Loading prepared data from {config.cache_file.name}...")
            # Upgrade compatible feature-only caches once, without ingestion or
            # encoder fitting. Later runs load the complete twelve-target data.
            if payload["data"].forecast_horizons != config.forecast_horizons:
                print("Adding forecast targets to the existing prepared cache...")
                payload["data"] = with_forecast_horizons(payload["data"], config)
                joblib.dump(payload, config.cache_file)
            return payload["data"]
        print("Prepared-data cache metadata is incompatible; rebuilding cache...")
    else:
        print("Preparing model data...")

    data = prepare_model_data(
        region=config.region,
        start=datetime.fromisoformat(config.start_date),
        end=datetime.fromisoformat(config.end_date),
        horizon=config.horizon_minutes,
        features=config.model_features,
        target=config.target_type,
        config=config.feature_config,
        learned_features=graph_embedding_stage(config, force_refresh=force_refresh),
    )
    data = with_forecast_horizons(data, config)
    payload = {
        "metadata": config.cache_metadata(),
        "data": data,
    }
    joblib.dump(payload, config.cache_file)
    print(f"Prepared data cached as {config.cache_file.name}.")
    return data


def chronological_validation_split(
    train_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological split with an embargo equal to the horizon."""
    split_index = int(len(train_df) * (1 - config.validation_fraction))
    if split_index <= 0 or split_index >= len(train_df):
        raise RuntimeError("Validation split produced an empty subset")

    validation_start = train_df.iloc[split_index]["settlement_date"]
    target_delta = pd.Timedelta(minutes=config.horizon_minutes)

    fit_df = train_df[
        (train_df["settlement_date"] + target_delta) < validation_start
    ].copy()
    validation_df = train_df[train_df["settlement_date"] >= validation_start].copy()

    if fit_df.empty or validation_df.empty:
        raise RuntimeError("Chronological validation split produced an empty subset")
    return fit_df, validation_df


def with_forecast_reliability(prepared: PreparedModelData) -> PreparedModelData:
    """Append the forecast reliability features to the model inputs."""
    extra = [
        column
        for column in forecast_reliability_columns(prepared.train_df)
        if column not in prepared.features
        and column in prepared.test_df.columns
        and prepared.train_df[column].notna().any()
    ]
    if not extra:
        raise ValueError(
            "Prepared data has no forecast reliability features; they need "
            "aligned AEMO P5MIN price forecasts"
        )
    features = [*prepared.features, *extra]
    return replace(
        prepared,
        X_train=prepared.train_df[features].copy(),
        X_test=prepared.test_df[features].copy(),
        features=features,
    )


def prepare_experiment(
    config: ExperimentConfig,
    *,
    force_refresh: bool = False,
) -> ExperimentData:
    """Load prepared data and construct the model-selection split."""
    if config.target_type == "aemo_error":
        features = replace(config.feature_config, p5min_horizons_minutes=config.forecast_horizons)
        config = replace(config, feature_config=features)
    prepared = load_prepared_data(config, force_refresh=force_refresh)
    if config.graph_embedding is not None:
        expected_embeddings = config.graph_embedding.embedding_columns
        if not set(expected_embeddings).issubset(prepared.features):
            raise ValueError("Prepared data must include all GNN embedding features; rebuild the cache")
    if config.forecast_reliability:
        prepared = with_forecast_reliability(prepared)
    # Explicit feature selections must not turn future outcomes into inputs.
    # Learned graph embeddings remain valid ordinary feature columns.
    leaked = [c for c in prepared.features if c == "target" or c.startswith("target_")]
    if leaked:
        raise ValueError(f"Target columns cannot be model inputs: {leaked}")
    fit_df, validation_df = chronological_validation_split(
        prepared.train_df,
        config,
    )
    experiment = ExperimentData(
        config=config,
        prepared=prepared,
        fit_df=fit_df,
        validation_df=validation_df,
        test_df=prepared.test_df,
    )
    check_embedding_precedes_validation(experiment)
    return experiment


def check_embedding_precedes_validation(experiment: ExperimentData) -> None:
    """Reject splits whose validation periods the graph encoder learned from.

    Validating a downstream model on labels the encoder was fitted to would
    make every candidate using the embedding look better than it is.
    """
    fit_end = getattr(experiment.prepared, "embedding_fit_end", None)
    if fit_end is None:
        return
    earliest = min(
        validation["settlement_date"].min()
        for _, validation in walk_forward_splits(experiment)
    )
    # fit_end bounds realised outcomes used for both fitting and checkpoint
    # selection. Subtracting the horizon would allow information from validation.
    if fit_end > earliest:
        raise ValueError(
            f"The graph encoder's information cutoff ({fit_end}) is "
            f"after the earliest validation block starts ({earliest}). Lower "
            "GraphEmbeddingConfig.fit_fraction or use fewer walk-forward folds."
        )


def with_forecast_horizons(prepared: PreparedModelData, config: ExperimentConfig) -> PreparedModelData:
    """Add exact-time labels after feature preparation, preserving learned features.

    Full timelines retain unlabeled rows as sequence context. Only supervised
    frames require every horizon, so all models are compared on the same origins.
    No fitted transforms or graph-encoder metadata are replaced here.
    """
    if prepared.train_timeline is None or prepared.test_timeline is None:
        raise ValueError("Multi-output experiments require full timelines; rerun with --refresh")
    test_start = prepared.test_timeline["settlement_date"].min()
    if prepared.train_timeline["settlement_date"].max() >= test_start:
        raise ValueError("Training and test timelines overlap")
    feature_config = replace(config.feature_config, target_horizons_minutes=config.horizons)
    targets = target_columns(config.horizons)
    timelines, supervised = [], []
    for is_training, timeline in ((True, prepared.train_timeline), (False, prepared.test_timeline)):
        if timeline.duplicated(["region", "settlement_date"]).any():
            raise ValueError("Duplicate region/timestamp rows make target alignment ambiguous")
        frame = add_targets(timeline, feature_config)
        for horizon, column in zip(config.horizons, targets):
            price = frame[f"target_rrp_h{horizon:02d}"]
            if config.target_type == "price":
                frame[column] = price
            else:
                base = "rrp" if config.target_type == "change" else f"p5_h{horizon:02d}_rrp"
                if base not in frame:
                    raise ValueError(f"Missing forecast input {base}; prepare all requested P5 horizons")
                frame[column] = price - frame[base]
        frame[targets] = frame[targets].replace([np.inf, -np.inf], np.nan)
        frame.loc[~np.isfinite(frame["rrp"]), targets] = np.nan
        if is_training:
            # The entire vector is embargoed at the latest forecast outcome.
            unresolved = frame["settlement_date"] + pd.Timedelta(minutes=config.horizon_minutes) >= test_start
            frame.loc[unresolved, targets] = np.nan
        frame["target"] = frame[targets[-1]]
        frame["target_rrp"] = frame[f"target_rrp_h{config.horizon_minutes:02d}"]
        rows = frame.dropna(subset=targets).reset_index(drop=True)
        if rows.empty:
            raise ValueError("No complete multi-horizon targets remain in a split")
        timelines.append(frame)
        supervised.append(rows)
    train, test = supervised
    return replace(
        prepared, forecast_horizons=config.forecast_horizons,
        train_timeline=timelines[0], test_timeline=timelines[1], train_df=train, test_df=test,
        X_train=train[prepared.features].copy(), X_test=test[prepared.features].copy(),
        y_train=train[targets].copy(), y_test=test[targets].copy(),
    )


def walk_forward_splits(experiment: ExperimentData):
    """Return expanding fit periods with disjoint validation blocks and label embargoes.

    Blocks are sized by eligible row count, not calendar duration. Later folds
    may train on earlier validation periods, as in periodic real-world retraining;
    the held-out test period is never included in these folds.
    """
    count = int(experiment.config.walk_forward_folds)
    if count == 1:
        return [(experiment.fit_df, experiment.validation_df)]
    frame = experiment.prepared.train_df
    width = len(frame) - int(len(frame) * (1 - experiment.config.validation_fraction))
    initial = len(frame) - count * width
    if initial <= 0:
        raise ValueError("Too many walk-forward folds for validation_fraction")
    folds = []
    for position in range(count):
        start = initial + position * width
        validation = frame.iloc[start : start + width].copy()
        cutoff = validation["settlement_date"].min()
        fit = frame.loc[
            frame["settlement_date"]
            + pd.Timedelta(minutes=experiment.config.horizon_minutes)
            < cutoff
        ].copy()
        if fit.empty:
            raise ValueError("Walk-forward fold has no eligible training rows")
        folds.append((fit, validation))
    return folds


def reconstruct_price_forecast(
    frame: pd.DataFrame,
    predicted_target: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Convert target-scale model output into an RRP forecast."""
    predicted_target = forecast_array(predicted_target, len(frame), len(config.horizons))
    if config.target_type == "price":
        return predicted_target
    if config.target_type == "change":
        base = frame["rrp"].to_numpy()[:, None]
    else:
        columns = [f"p5_h{h:02d}_rrp" for h in config.horizons]
        base = frame[columns].to_numpy()
    return base + predicted_target


def print_experiment_summary(experiment: ExperimentData) -> None:
    """Report the prepared sample, features, and chronological split."""
    data = experiment.prepared
    config = experiment.config
    train_df = data.train_df
    test_df = data.test_df

    print("\nPREPARED MODEL DATA")
    print("-------------------")
    print("Target region:", data.region)
    print("Forecast horizons:", config.horizons, "minutes")
    print("Target type:", data.target_type)
    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Features:", len(data.features))
    print("P5MIN features:", sum(c.startswith("p5_h") for c in data.features))
    print("Interconnector features:", sum(c.startswith("ic_") for c in data.features))
    print(
        "Training period:",
        train_df["settlement_date"].min(),
        "->",
        train_df["settlement_date"].max(),
    )
    print(
        "Test period:",
        test_df["settlement_date"].min(),
        "->",
        test_df["settlement_date"].max(),
    )

    if data.embedding_columns:
        print("Graph embedding features:", len(data.embedding_columns))
        print("Graph encoder fitted before:", data.embedding_fit_end)

    column = config.aemo_forecast_column
    print("\nAEMO FORECAST AVAILABILITY")
    print("--------------------------")
    for label, frame in (("training", train_df), ("test", test_df)):
        count = frame[column].notna().sum() if column in frame.columns else 0
        print(f"Valid {label} {column}:", count)

    print("\nCHRONOLOGICAL VALIDATION")
    print("------------------------")
    print(
        "Fit:",
        experiment.fit_df["settlement_date"].min(),
        "->",
        experiment.fit_df["settlement_date"].max(),
    )
    print(
        "Validation:",
        experiment.validation_df["settlement_date"].min(),
        "->",
        experiment.validation_df["settlement_date"].max(),
    )
    print("Fit rows:", len(experiment.fit_df))
    print("Validation rows:", len(experiment.validation_df))
