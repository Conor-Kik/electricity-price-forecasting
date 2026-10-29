"""Saved runs replay the same test inputs without preparation, tuning or fitting."""

import hashlib
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import main
import pipeline.experiment_pipeline as preparation
from electricity_price_forecasting_model.baseline_models.arima import ARIMAModel
from electricity_price_forecasting_model.baseline_models.xgboost import XGBoostModel
from electricity_price_forecasting_model.final_model.gru import GRUModel
from electricity_price_forecasting_model.final_model.mlp import MLPModel
from pipeline.artifacts import create_run_directory, save_model_run
from pipeline.contracts import price_values
from pipeline.evaluation import evaluate
from pipeline.experiment_pipeline import ExperimentConfig, HOUR_HORIZONS, reconstruct_price_forecast, with_forecast_horizons
from pipeline.model_data import final_training_data, predict_test
from pipeline.graph_embedding.embedder import GraphEmbedder
from pipeline.preprocessing import FeatureConfig, prepare_model_data
from pipeline.saved_runs import load_saved_run


@pytest.fixture
def saved_run_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "source_fingerprint", lambda: "test-source")
    monkeypatch.setattr(preparation, "graph_source_fingerprint", lambda: "test-graph")

    def make(kind="mlp", graph=False, target="change", reliability=False):
        raw = pd.DataFrame({
            "region": "NSW1",
            "settlement_date": pd.date_range("2024-01-01", periods=140, freq="5min"),
            "rrp": 100 + np.sin(np.arange(140)),
            "totaldemand": 200 + np.arange(140),
            "availablegeneration": 500.0,
        })
        config = ExperimentConfig(
            model_features=["rrp", "totaldemand"], target_type=target,
            cache_directory=tmp_path / "cache", random_seed=17,
            forecast_reliability=reliability,
            feature_config=FeatureConfig(price_lags=(1, 3), rolling_windows=(2,)),
            graph_embedding=preparation.GraphEmbeddingConfig(node_features=("rrp",)) if graph else None,
        )
        prepared = prepare_model_data(
            horizon=60, features=config.model_features, target="change", config=config.feature_config,
            train_data=SimpleNamespace(df=raw.iloc[:100]),
            test_data=SimpleNamespace(df=raw.iloc[100:]),
        )
        if graph:
            columns = config.graph_embedding.embedding_columns
            for timeline in (prepared.train_timeline, prepared.test_timeline):
                timeline[columns] = 0.0
            prepared = replace(prepared, features=[*prepared.features, *columns], embedding_columns=columns)
        if target == "aemo_error":
            for timeline in (prepared.train_timeline, prepared.test_timeline):
                for h in HOUR_HORIZONS:
                    timeline[f"p5_h{h:02d}_rrp"] = timeline["rrp"]
        if reliability:
            for timeline in (prepared.train_timeline, prepared.test_timeline):
                timeline["reliability_h05_mae_12"] = 1.0
        prepared = replace(prepared, target_type=target)
        prepared = with_forecast_horizons(prepared, config)
        config.cache_directory.mkdir(exist_ok=True)
        joblib.dump({"metadata": config.cache_metadata(), "data": prepared}, config.cache_file)
        if reliability:
            prepared = preparation.with_forecast_reliability(prepared)
        models = {
            "mlp": lambda: MLPModel(hidden_sizes=(4,), dropout=0, max_epochs=1, device="cpu"),
            "gru": lambda: GRUModel(sequence_length=3, hidden_size=4, max_epochs=1, device="cpu"),
            "xgboost": lambda: XGBoostModel(max_estimators=2, n_jobs=1, forecast_horizons=HOUR_HORIZONS),
            "arima": lambda: ARIMAModel(p=0, d=0, q=0, daily_harmonics=0, fit_window_days=None,
                                        horizon_minutes=60, target_type=target, forecast_horizons=HOUR_HORIZONS),
        }
        model = models[kind]()
        model.show_progress = False
        model.train(*final_training_data(model, prepared))
        predictions = reconstruct_price_forecast(prepared.test_df, predict_test(model, prepared), config)
        result = evaluate(model, price_values(prepared.test_df, HOUR_HORIZONS), predictions)
        directory = create_run_directory(config, tmp_path / "runs", prepared)
        search = SimpleNamespace(trials=pd.DataFrame({"score": [0.0]}), folds=pd.DataFrame({"score": [0.0]}),
                                 best_config={}, refit_config={}, metric="mae", best_score=0.0)
        save_model_run(directory, model, search, prepared.test_df, predictions, result,
                       prepared=prepared, graph_embedding=config.graph_embedding)
        return directory, config, prepared, predictions

    return make


def forbid_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Saved-run evaluation attempted data preparation or model fitting")

    monkeypatch.setattr(main, "prepare_experiment", forbidden)
    monkeypatch.setattr(main, "search_hyperparameters", forbidden)
    monkeypatch.setattr(preparation, "prepare_model_data", forbidden)
    monkeypatch.setattr(preparation, "graph_embedding_stage", forbidden)
    monkeypatch.setattr(GraphEmbedder, "fit", forbidden)
    for model_type in (MLPModel, GRUModel, XGBoostModel, ARIMAModel):
        monkeypatch.setattr(model_type, "train", forbidden)
        monkeypatch.setattr(model_type, "train_with_validation", forbidden)


def file_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind,graph,target", [
    ("mlp", False, "price"), ("gru", False, "change"), ("gru", True, "aemo_error"),
    ("xgboost", False, "change"), ("xgboost", True, "change"), ("arima", False, "change"),
])
def test_load_and_cli_reproduce_forecasts_without_fitting(saved_run_factory, monkeypatch, kind, graph, target):
    directory, config, prepared, expected = saved_run_factory(kind, graph, target)
    before = file_hashes(directory)
    cache_before = config.cache_file.read_bytes()
    forbid_training(monkeypatch)
    loaded = load_saved_run(directory, [kind.upper()])
    assert loaded.config == config
    assert loaded.config.feature_config.price_lags == (1, 3)
    model = loaded.models[kind].model
    actual = reconstruct_price_forecast(prepared.test_df, predict_test(model, loaded.prepared), loaded.config)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    observed = []
    original = main.evaluate

    def capture(model, actual, predicted):
        observed.append(predicted)
        return original(model, actual, predicted)

    monkeypatch.setattr(main, "evaluate", capture)
    # The saved experiment takes precedence over today's defaults and grids.
    monkeypatch.setattr(main, "EXPERIMENT", replace(config, region="VIC1"))
    main.run_cli(["--load-run", str(directory), "--models", kind, "--skip-shap", "--device", "cpu"])
    np.testing.assert_allclose(observed[0], expected, rtol=1e-6)
    assert file_hashes(directory) == before
    assert config.cache_file.read_bytes() == cache_before


def test_loaded_run_can_export_new_reports_and_shap(saved_run_factory, monkeypatch, tmp_path):
    directory, config, _, expected = saved_run_factory("gru", graph=True)
    before = file_hashes(directory)
    forbid_training(monkeypatch)
    main.run_cli(["--load-run", str(directory), "--save", "--output-dir", str(tmp_path / "evaluations"),
                  "--device", "cpu", "--shap-horizons", "5", "60", "--shap-samples", "2", "--shap-draws", "2"])
    output = next((tmp_path / "evaluations").iterdir())
    manifest = json.loads((output / "experiment.json").read_text())
    assert manifest["source_run"] == str(directory.resolve())
    assert (output / "leaderboard.csv").is_file()
    assert (output / "errors_by_horizon.png").is_file()
    assert (output / "gru" / "evaluation.json").is_file()
    assert not (output / "gru" / "model.joblib").exists()
    assert (output / "gru" / "gradient_shap_h05.csv").is_file()
    assert (output / "gru" / "gradient_shap_feature_sequence_h60.png").is_file()
    saved = pd.read_csv(output / "gru" / "predictions.csv")
    np.testing.assert_allclose(saved["prediction_rrp"], expected.T.reshape(-1), rtol=1e-6)
    assert file_hashes(directory) == before
    with pytest.raises(ValueError, match="original source_run"):
        load_saved_run(output)


@pytest.mark.parametrize("problem,match", [
    ("missing-cache", "cache is missing"), ("cache-metadata", "incompatible"),
    ("source", "provenance"), ("horizons", "horizons"),
    ("features", "feature order"), ("gnn", "GNN embedding"),
    ("target", "prices/targets"), ("origins", "origins"),
])
def test_incompatible_data_fails_without_rebuilding(saved_run_factory, monkeypatch, problem, match):
    directory, config, _, _ = saved_run_factory(graph=True)
    payload = joblib.load(config.cache_file)
    if problem == "missing-cache":
        config.cache_file.unlink()
    elif problem == "source":
        monkeypatch.setattr(preparation, "source_fingerprint", lambda: "changed-source")
    else:
        if problem == "cache-metadata":
            payload["metadata"]["schema_version"] = -1
        elif problem == "horizons":
            payload["data"].forecast_horizons = (5,)
        elif problem == "features":
            payload["data"].features = list(reversed(payload["data"].features))
        elif problem == "gnn":
            payload["data"].features = ["rrp", "totaldemand"]
        elif problem == "target":
            payload["data"].test_df.loc[0, "target_h05"] += 1
        elif problem == "origins":
            payload["data"].test_df.loc[0, "settlement_date"] += pd.Timedelta(minutes=1)
        joblib.dump(payload, config.cache_file)
    forbid_training(monkeypatch)
    with pytest.raises(ValueError, match=match):
        main.main(load_run=directory, skip_shap=True)


def test_missing_manifest_and_unsaved_model_are_clear_errors(saved_run_factory, tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        load_saved_run(tmp_path)
    directory, *_ = saved_run_factory()
    with pytest.raises(ValueError, match="not saved.*lstm"):
        load_saved_run(directory, ["lstm"])


def test_loads_all_saved_models_independently_of_current_registry(saved_run_factory, monkeypatch):
    directory, *_ = saved_run_factory("gru")
    other, *_ = saved_run_factory("mlp")
    shutil.copytree(other / "mlp", directory / "mlp")
    forbid_training(monkeypatch)
    monkeypatch.setattr(main, "MODEL_REGISTRY", [])
    names = []
    original = main.evaluate

    def capture(model, actual, predicted):
        names.append(model.get_model_name())
        return original(model, actual, predicted)

    monkeypatch.setattr(main, "evaluate", capture)
    main.run_cli(["--load-run", str(directory), "--skip-shap", "--device", "cpu"])
    assert names == ["GRU", "MLP"]
    assert list(load_saved_run(directory, ["mlp"]).models) == ["mlp"]


def test_restores_optional_reliability_features_without_updating_cache(saved_run_factory, monkeypatch):
    directory, config, prepared, expected = saved_run_factory(reliability=True)
    cache_before = config.cache_file.read_bytes()
    forbid_training(monkeypatch)
    loaded = load_saved_run(directory)
    assert loaded.config.forecast_reliability
    assert loaded.prepared.features == prepared.features
    actual = reconstruct_price_forecast(loaded.prepared.test_df,
                                       predict_test(loaded.models["mlp"].model, loaded.prepared), loaded.config)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert config.cache_file.read_bytes() == cache_before


def test_plot_seed_changes_examples_without_changing_forecasts(saved_run_factory, monkeypatch):
    import matplotlib.pyplot as plt
    import pipeline.plotting as plotting

    directory, config, _, expected = saved_run_factory()
    before = file_hashes(directory)
    forbid_training(monkeypatch)
    samples, seeds = [], []

    def capture(results, prepared, importances, *, seed, **kwargs):
        seeds.append(seed)
        np.testing.assert_allclose(results[0].predictions, expected, rtol=1e-6)
        figure = plotting.plot_forecast_examples(results, prepared, seed=seed)
        samples.append([axis.get_title() for axis in figure.axes])
        plt.close(figure)

    monkeypatch.setattr(plotting, "present_results", capture)
    command = ["--load-run", str(directory), "--skip-shap", "--plot", "--device", "cpu"]
    main.run_cli(command)
    main.run_cli([*command, "--plot-seed", "7"])
    main.run_cli([*command, "--plot-seed", "7"])
    assert seeds == [config.random_seed, 7, 7]
    assert samples[0] != samples[1]
    assert samples[1] == samples[2]
    assert file_hashes(directory) == before


def test_plot_seed_rejects_negative_values():
    with pytest.raises(SystemExit):
        main.parse_args(["--plot-seed", "-1"])


@pytest.mark.parametrize("option", [["--refresh"], ["--epochs=1"], ["--folds", "1"],
                                   ["--metric", "mae"], ["--seed", "42"]])
def test_load_cli_rejects_training_overrides(option):
    with pytest.raises(SystemExit):
        main.parse_args(["--load-run", "saved-run", *option])
