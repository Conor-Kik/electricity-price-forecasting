"""Regression coverage for temporal alignment and shared model integration."""

import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from electricity_price_forecasting_model.final_model.gru import GRUModel
from electricity_price_forecasting_model.final_model.mlp import MLPModel
from pipeline.contracts import indexed_features, validate_timeline
from pipeline.evaluation import evaluate
from pipeline.experiment_pipeline import (
    PROJECT_ROOT,
    ExperimentConfig,
    ExperimentData,
    _cache_payload_is_valid,
    chronological_validation_split,
    reconstruct_price_forecast,
    walk_forward_splits,
    with_forecast_horizons,
)
from pipeline.model_data import final_training_data, predict_test, selection_data
from pipeline.preprocessing import FeatureConfig, make_sequences, prepare_model_data
from pipeline.tuning import search_hyperparameters


@pytest.fixture
def experiment():
    count = 200
    raw = pd.DataFrame(
        {
            "region": ["NSW1"] * count,
            "settlement_date": pd.date_range("2024-01-01", periods=count, freq="5min"),
            "rrp": 100 + np.sin(np.arange(count)),
            "totaldemand": 200 + np.arange(count),
            "availablegeneration": np.full(count, 500.0),
        }
    )
    features = ["rrp", "totaldemand"]
    feature_config = FeatureConfig(
        price_lags=(1,), demand_lags=(1,), rolling_windows=(2,)
    )
    prepared = prepare_model_data(
        horizon=60,
        features=features,
        config=feature_config,
        train_data=SimpleNamespace(df=raw.iloc[:160]),
        test_data=SimpleNamespace(df=raw.iloc[160:]),
    )
    config = ExperimentConfig(
        validation_fraction=0.2,
        model_features=features,
        feature_config=feature_config,
    )
    # Fixed synthetic embeddings isolate downstream alignment/training tests.
    columns = config.graph_embedding.embedding_columns
    for frame in (prepared.train_timeline, prepared.test_timeline, prepared.train_df, prepared.test_df,
                  prepared.X_train, prepared.X_test):
        frame[columns] = 0.0
    prepared = replace(prepared, features=[*prepared.features, *columns], embedding_columns=columns)
    prepared = with_forecast_horizons(prepared, config)
    fit, validation = chronological_validation_split(prepared.train_df, config)
    return ExperimentData(config, prepared, fit, validation, prepared.test_df)


def small_gru(**kwargs):
    return GRUModel(
        sequence_length=3,
        hidden_size=4,
        max_epochs=1,
        batch_size=8,
        device="cpu",
        **kwargs,
    )


def test_timeline_retains_unlabeled_training_tail(experiment):
    prepared = experiment.prepared
    assert len(prepared.train_timeline) == 160
    assert len(prepared.train_df) == 148
    assert prepared.train_timeline["target"].tail(12).isna().all()
    assert len(prepared.test_timeline) == 40
    assert len(prepared.test_df) == 28


def test_embargo_masks_labels_but_retains_recent_context(experiment):
    model = small_gru()
    data = selection_data(model, experiment)
    times = data.x_train.index.get_level_values("settlement_date")
    start = data.x_validation.index.get_level_values("settlement_date")[0]
    assert times[-1] == start - pd.Timedelta(minutes=5)
    assert data.y_train.tail(12).isna().all().all()
    assert data.y_train.notna().all(axis=1).sum() == len(experiment.fit_df)
    model.train_with_validation(
        data.x_train, data.y_train, data.x_validation, data.y_validation
    )
    assert model.context_frame.index[-1] == data.x_train.index[-1]
    # The scaler does not fit on embargo rows.
    pd.testing.assert_series_equal(
        model.feature_medians, data.x_train.loc[data.y_train.notna().all(axis=1)].median()
    )


def test_explicit_context_is_invariant_to_prediction_chunks(experiment):
    model = small_gru()
    x_train, y_train = final_training_data(model, experiment.prepared)
    model.train(x_train, y_train)
    current = indexed_features(experiment.prepared.test_timeline, experiment.features)
    full = model.predict(current, history=x_train)
    changed_future = current.copy()
    changed_future.iloc[4:] = 9999.0
    np.testing.assert_allclose(
        model.predict(changed_future, history=x_train)[:4],
        full[:4],
        rtol=1e-5,
        atol=1e-5,
    )
    first = model.predict(current.iloc[:4], history=x_train)
    history = pd.concat([x_train, current.iloc[:4]])
    second = model.predict(current.iloc[4:], history=history)
    np.testing.assert_allclose(
        full, np.concatenate([first, second]), rtol=1e-5, atol=1e-5
    )
    with pytest.raises(ValueError, match="gaps"):
        model.predict(current.iloc[4:])  # stale fallback cannot bridge the time gap
    assert len(predict_test(model, experiment.prepared)) == len(experiment.test_df)


def test_tcn_runs_through_selection_refit_and_test_alignment(experiment):
    name = "electricity_price_forecasting_model.final_model.tcn"
    assert importlib.util.find_spec(name) is not None, "TCN model module is missing"
    TCNModel = importlib.import_module(name).TCNModel
    grid = {
        "sequence_length": [3],
        "channels": [4],
        "kernel_size": [3],
        "num_blocks": [1],
        "dropout": [0.0],
        "batch_size": [8],
        "max_epochs": [1],
        "device": ["cpu"],
    }

    search = search_hyperparameters(TCNModel, grid, experiment, verbose=False)
    model = TCNModel(**search.best_config)
    model.train(*final_training_data(model, experiment.prepared))
    predictions = predict_test(model, experiment.prepared)
    prices = reconstruct_price_forecast(
        experiment.test_df, predictions, experiment.config
    )
    result = evaluate(model, experiment.test_df[[f"target_rrp_h{h:02d}" for h in experiment.config.horizons]].to_numpy(), prices)

    assert search.best_config["sequence_length"] == 3
    assert len(predictions) == len(experiment.test_df)
    assert np.isfinite(predictions).all()
    assert np.isfinite(result.metrics["mae"])


@pytest.mark.parametrize("problem", ["gap", "reverse", "duplicate", "region"])
def test_temporal_metadata_rejects_invalid_sequences(problem):
    frame = pd.DataFrame(
        {
            "region": ["NSW1"] * 3,
            "settlement_date": pd.date_range("2024-01-01", periods=3, freq="5min"),
        }
    )
    if problem == "gap":
        frame = frame.iloc[[0, 2]]
    elif problem == "reverse":
        frame = frame.iloc[::-1]
    elif problem == "duplicate":
        frame.loc[2, "settlement_date"] = frame.loc[1, "settlement_date"]
    else:
        frame.loc[2, "region"] = "QLD1"
    with pytest.raises(ValueError):
        validate_timeline(frame)


def test_eager_helper_rejects_future_history(experiment):
    frame = experiment.prepared.train_timeline
    with pytest.raises(ValueError, match="strictly precede"):
        make_sequences(
            frame.iloc[:2],
            experiment.features,
            target_columns=["target"],
            sequence_length=3,
            history=frame.iloc[3:6],
        )


def test_forecasts_require_all_twelve_horizons():
    frame = pd.DataFrame({"rrp": [10.0, 20.0]})
    config = ExperimentConfig()
    predicted = reconstruct_price_forecast(frame, np.ones((2, 12)), config)
    np.testing.assert_array_equal(predicted, np.tile([[11.0], [21.0]], (1, 12)))
    for invalid in (np.ones(2), np.ones((2, 1)), np.ones((2, 2))):
        with pytest.raises(ValueError, match="shape"):
            reconstruct_price_forecast(frame, invalid, config)


def test_cache_metadata_tracks_feature_config_and_source(experiment, monkeypatch):
    import pipeline.experiment_pipeline as module

    monkeypatch.setattr(module, "source_fingerprint", lambda: "source-a")
    config = experiment.config
    payload = {"metadata": config.cache_metadata(), "data": experiment.prepared}
    assert _cache_payload_is_valid(payload, config)
    changed = replace(
        config, feature_config=replace(config.feature_config, price_lags=(2,))
    )
    assert not _cache_payload_is_valid(payload, changed)
    monkeypatch.setattr(module, "source_fingerprint", lambda: "source-b")
    assert not _cache_payload_is_valid(payload, config)
    # Encoder changes invalidate every experiment cache.
    monkeypatch.setattr(module, "source_fingerprint", lambda: "source-a")
    monkeypatch.setattr(module, "graph_source_fingerprint", lambda: "graph-b")
    assert not _cache_payload_is_valid(payload, config)
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_walk_forward_folds_are_disjoint_and_embargoed(experiment):
    experiment = replace(
        experiment, config=replace(experiment.config, walk_forward_folds=3)
    )
    splits = walk_forward_splits(experiment)
    assert len(splits) == 3
    for fit, validation in splits:
        assert (
            fit["settlement_date"] + pd.Timedelta(minutes=60)
            < validation["settlement_date"].min()
        ).all()
    assert splits[0][1]["settlement_date"].max() < splits[1][1]["settlement_date"].min()
    assert len(splits[0][0]) < len(splits[1][0]) < len(splits[2][0])


def test_search_injects_seed_metric_and_refit_even_without_epoch_grid(
    experiment, monkeypatch
):
    # Network training itself is covered above; this isolates selection/refit wiring.
    observed = []

    def fake_train(self, x, y, xv, yv):
        observed.append((self.random_seed, self.selection_metric, len(x)))
        return SimpleNamespace(best_epoch=2, refit_config={"max_epochs": 2})

    monkeypatch.setattr(MLPModel, "train_with_validation", fake_train)
    monkeypatch.setattr(MLPModel, "predict", lambda self, x: np.zeros((len(x), 12)))
    result = search_hyperparameters(
        MLPModel,
        {"hidden_sizes": [(4,)], "dropout": [0]},
        replace(experiment, config=replace(experiment.config, walk_forward_folds=2)),
        metric="rmse",
        seed=17,
        verbose=False,
    )
    assert len(observed) == 2
    assert all(seed == 17 and metric == "rmse" for seed, metric, _ in observed)
    assert result.best_config["max_epochs"] == 2
    assert result.refit_config == {"max_epochs": 2}
    assert result.best_config["random_seed"] == 17
    assert len(result.folds) == 2


def test_shared_validation_metric_is_used(experiment, monkeypatch):
    model = small_gru(selection_metric="rmse")
    data = selection_data(model, experiment)
    calls = []
    monkeypatch.setattr(
        model, "_validation_score", lambda x, y, metric: calls.append(metric) or 2.0
    )
    monkeypatch.setattr(
        model,
        "_validation_mae",
        lambda *args: pytest.fail("MAE selected during RMSE run"),
    )
    model.train_with_validation(
        data.x_train, data.y_train, data.x_validation, data.y_validation
    )
    assert calls == ["rmse"]
    assert model.validation_score_history == [2.0]
    assert model.validation_mae_history == []


@pytest.mark.parametrize("use_gnn", [True, False])
def test_saved_run_round_trips_model_and_context(
    experiment, tmp_path, monkeypatch, use_gnn
):
    import main
    import matplotlib

    matplotlib.use("Agg", force=True)

    if not use_gnn:
        prepared = experiment.prepared
        features = [c for c in prepared.features if c not in prepared.embedding_columns]
        prepared = replace(prepared, features=features, embedding_columns=[],
                           X_train=prepared.X_train[features], X_test=prepared.X_test[features])
        experiment = replace(experiment, config=replace(experiment.config, graph_embedding=None),
                             prepared=prepared)
    model_type = GRUModel
    monkeypatch.setattr(main, "prepare_experiment", lambda config, **kwargs: experiment)
    grid = {
        "sequence_length": [3],
        "hidden_size": [4],
        "max_epochs": [1],
        "batch_size": [8],
        "device": ["cpu"],
    }
    monkeypatch.setattr(main, "MODEL_REGISTRY", [(model_type, grid)])
    main.main(config=experiment.config, output_directory=tmp_path, save_results=True, skip_shap=True)
    directory = next(tmp_path.iterdir())
    assert (directory / "experiment.json").exists()
    assert (directory / "summary.csv").exists()
    model_directory = directory / model_type.model_name.lower()
    for filename in (
        "trials.csv",
        "folds.csv",
        "model.json",
        "predictions.csv",
        "model.joblib",
    ):
        assert (model_directory / filename).exists()
    metadata = json.loads((model_directory / "model.json").read_text())
    if use_gnn:
        assert metadata["graph_embedding"]["adjacency_mode"] == "nem"
    else:
        assert metadata["graph_embedding"] is None
    saved = pd.read_csv(model_directory / "predictions.csv")
    model = joblib.load(model_directory / "model.joblib")
    prediction = reconstruct_price_forecast(
        experiment.test_df, predict_test(model, experiment.prepared), experiment.config
    )
    np.testing.assert_allclose(prediction.T.reshape(-1), saved["prediction_rrp"], rtol=1e-6)
    assert model.device.type == "cpu"


def test_shared_gradient_clipping_and_scheduler(experiment, monkeypatch):
    import torch

    clip_calls, learning_rates = [], []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = torch.optim.lr_scheduler.StepLR.step

    def clip(parameters, max_norm, *args, **kwargs):
        clip_calls.append(max_norm)
        return original_clip(parameters, max_norm, *args, **kwargs)

    def step(scheduler, *args, **kwargs):
        result = original_step(scheduler, *args, **kwargs)
        learning_rates.append(scheduler.optimizer.param_groups[0]["lr"])
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(torch.optim.lr_scheduler.StepLR, "step", step)
    model = GRUModel(
        sequence_length=3,
        hidden_size=4,
        max_epochs=2,
        batch_size=8,
        learning_rate=0.01,
        gradient_clip_norm=0.5,
        scheduler_step_size=1,
        scheduler_gamma=0.5,
        device="cpu",
    )
    model.train(*final_training_data(model, experiment.prepared))
    assert clip_calls and all(value == 0.5 for value in clip_calls)
    assert learning_rates[-1] == pytest.approx(0.0025)
