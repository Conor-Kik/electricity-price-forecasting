"""Load trusted saved forecasters and their existing data without fitting anything."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from electricity_price_forecasting_model.baseline_models.xgboost import XGBoostModel
from pipeline.contracts import target_columns
from pipeline.experiment_pipeline import (
    ExperimentConfig, HOUR_HORIZONS, with_forecast_reliability,
)
from pipeline.graph_embedding import GraphEmbeddingConfig
from pipeline.model import Model
from pipeline.preprocessing import FeatureConfig, PreparedModelData


def _json_value(value):
    """Compare JSON manifests with cache metadata containing Python tuples."""
    return json.loads(json.dumps(value, default=str))


def _restore_config(values):
    values = dict(values)
    values["cache_directory"] = Path(values["cache_directory"])
    feature_values = dict(values["feature_config"])
    for item in fields(FeatureConfig):
        if isinstance(item.default, tuple) and item.name in feature_values:
            feature_values[item.name] = tuple(feature_values[item.name])
    values["feature_config"] = FeatureConfig(**feature_values)
    if values["graph_embedding"] is not None:
        graph = dict(values["graph_embedding"])
        graph["regions"] = tuple(graph["regions"])
        if graph["node_features"] is not None:
            graph["node_features"] = tuple(graph["node_features"])
        values["graph_embedding"] = GraphEmbeddingConfig(**graph)
    return ExperimentConfig(**values)


@dataclass
class SavedModel:
    directory: Path
    model: Model
    metadata: dict


@dataclass
class SavedRun:
    directory: Path
    config: ExperimentConfig
    prepared: PreparedModelData
    models: dict[str, SavedModel]


def _check_test_rows(directory, prepared):
    """Reject changed origins or labels, even if a cache has matching metadata."""
    path = directory / "predictions.csv"
    if not path.is_file():
        raise ValueError(f"Missing saved test-origin reference: {path}")
    columns = ["region", "settlement_date", "horizon_minutes", "rrp", "target_rrp", "target"]
    saved = pd.read_csv(path, usecols=columns, parse_dates=["settlement_date"])
    frame = prepared.test_df.reset_index(drop=True)
    if len(saved) != len(frame) * len(HOUR_HORIZONS) or set(saved["horizon_minutes"]) != set(HOUR_HORIZONS):
        raise ValueError("Saved forecast origins/horizons do not match the prepared test data")
    for horizon in HOUR_HORIZONS:
        rows = saved.loc[saved["horizon_minutes"] == horizon].reset_index(drop=True)
        expected = frame
        if (len(rows) != len(expected)
                or not rows["region"].equals(expected["region"])
                or not rows["settlement_date"].equals(expected["settlement_date"])):
            raise ValueError("Saved forecast origins do not match the prepared test data")
        actual_columns = ["rrp", f"target_rrp_h{horizon:02d}", f"target_h{horizon:02d}"]
        if not np.allclose(rows[["rrp", "target_rrp", "target"]].to_numpy(dtype=float),
                           expected[actual_columns].to_numpy(dtype=float),
                           rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError("Saved test prices/targets do not match the prepared test data")


def load_saved_run(directory, selected_models=None) -> SavedRun:
    """Read only existing trusted artifacts; never ingest, prepare, upgrade or fit.

    Matching prepared data is required, including cached GNN embeddings for
    graph-enabled runs. Joblib files must come from a trusted source.
    """
    directory = Path(directory).resolve()
    manifest_path = directory / "experiment.json"
    if not manifest_path.is_file():
        raise ValueError(f"No saved experiment manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "source_run" in manifest:
        raise ValueError("This directory contains evaluation reports; load its original source_run instead")
    config = _restore_config(manifest["experiment"])
    metadata = _json_value(config.cache_metadata())
    if manifest["data_metadata"] != metadata:
        raise ValueError("Saved-run data provenance differs from the current preparation code or raw caches")
    if not config.cache_file.is_file():
        raise ValueError(f"Prepared-data cache is missing: {config.cache_file}. Restore it; --load-run never rebuilds data or GNN features")
    payload = joblib.load(config.cache_file)
    if (not isinstance(payload, dict) or _json_value(payload.get("metadata")) != metadata
            or not isinstance(payload.get("data"), PreparedModelData)):
        raise ValueError("Prepared-data cache is incompatible with the saved run")
    prepared = payload["data"]
    if (prepared.region != config.region or prepared.target_type != config.target_type
            or prepared.forecast_horizons != HOUR_HORIZONS):
        raise ValueError("Prepared-data region, target or horizons do not match the saved run")
    if config.forecast_reliability:
        prepared = with_forecast_reliability(prepared)
    if config.graph_embedding is not None:
        expected = set(config.graph_embedding.embedding_columns)
        if not expected.issubset(prepared.features):
            raise ValueError("Prepared-data cache is missing the saved run's GNN embedding inputs")
    available = {path.parent.name: path.parent for path in directory.glob("*/model.joblib")}
    requested = {name.lower() for name in selected_models} if selected_models else set(available)
    if not requested:
        raise ValueError("No fitted model bundles were found in this run")
    if requested - available.keys():
        raise ValueError(f"Models not saved in this run: {', '.join(sorted(requested - available.keys()))}")
    models = {}
    for slug in sorted(requested):
        folder = available[slug]
        info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        if info["graph_embedding"] != _json_value(manifest["experiment"]["graph_embedding"]):
            raise ValueError(f"{slug}: graph configuration differs from the saved experiment")
        model = joblib.load(folder / "model.joblib")
        if model.get_model_name() != info["model"] or info["model"].lower().replace(" ", "-") != slug:
            raise ValueError(f"{slug}: fitted model identity does not match its metadata")
        features = list(getattr(model, "feature_columns", []))
        if features != info["features"]:
            raise ValueError(f"{slug}: fitted model features do not match its metadata")
        if isinstance(model, XGBoostModel):
            features = [name for name in features if name != "horizon_minutes"]
        # ARIMA reads named price columns from temporal inputs, not feature_columns.
        if features and features != prepared.features:
            raise ValueError(f"{slug}: model inputs differ from the prepared feature order")
        targets = getattr(model, "target_columns", None)
        if targets != info.get("target_columns"):
            raise ValueError(f"{slug}: fitted targets do not match the model metadata")
        if targets is not None and list(targets) != target_columns(HOUR_HORIZONS):
            raise ValueError(f"{slug}: fitted target columns do not match the twelve saved horizons")
        if tuple(getattr(model, "forecast_horizons", HOUR_HORIZONS)) != HOUR_HORIZONS:
            raise ValueError(f"{slug}: fitted forecast horizons do not match the experiment")
        if getattr(model, "output_size", len(HOUR_HORIZONS)) != len(HOUR_HORIZONS):
            raise ValueError(f"{slug}: fitted model does not have twelve outputs")
        if (getattr(model, "target_type", config.target_type) != config.target_type
                or getattr(model, "interval_minutes", config.feature_config.interval_minutes)
                != config.feature_config.interval_minutes):
            raise ValueError(f"{slug}: fitted target type or input interval differs from the experiment")
        _check_test_rows(folder, prepared)
        models[slug] = SavedModel(folder, model, info)
    return SavedRun(directory, config, prepared, models)
