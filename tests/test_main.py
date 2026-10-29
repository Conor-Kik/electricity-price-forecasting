from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("main_under_test", MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_importing_main_does_not_start_an_experiment(monkeypatch):
    from pipeline import experiment_pipeline

    def unexpected_run(*args, **kwargs):
        raise AssertionError("importing main.py started the experiment")

    monkeypatch.setattr(
        experiment_pipeline,
        "prepare_experiment",
        unexpected_run,
    )

    load_main_module()


def test_cli_selects_requested_models_in_registry_order():
    module = load_main_module()

    selected = module.select_model_registry(["mlp"])

    assert [factory.model_name for factory, _ in selected] == ["MLP"]


def test_cli_selects_gru():
    module = load_main_module()

    selected = module.select_model_registry(["gru"])

    assert [factory.model_name for factory, _ in selected] == ["GRU"]


def test_cli_selects_lstm():
    module = load_main_module()

    selected = module.select_model_registry(["lstm"])

    assert [factory.model_name for factory, _ in selected] == ["LSTM"]


def test_cli_selects_arima_without_duplicating_experiment_settings():
    module = load_main_module()

    selected = module.select_model_registry(["arima"])

    assert [factory.model_name for factory, _ in selected] == ["ARIMA"]
    grid = selected[0][1]
    # Tuning supplies these from the active experiment for every candidate.
    assert {"horizon_minutes", "target_type", "forecast_horizons"}.isdisjoint(grid)


def test_cli_selects_tcn_with_the_initial_comparison_window():
    module = load_main_module()

    assert "tcn" in [module.model_cli_name(factory) for factory, _ in module.MODEL_REGISTRY]
    selected = module.select_model_registry(["tcn"])

    assert [factory.model_name for factory, _ in selected] == ["TCN"]
    grid = selected[0][1]
    assert grid["sequence_length"] == [12]
    assert grid["channels"] == [16]
    assert grid["kernel_size"] == [3]
    assert grid["num_blocks"] == [2]
    assert module.parse_args(["--models", "tcn"]).models == ["tcn"]


def test_cli_defaults_to_all_registered_models():
    module = load_main_module()

    args = module.parse_args([])

    assert args.models is None
    assert module.select_model_registry(args.models) == module.MODEL_REGISTRY


def test_cli_accepts_uppercase_tcn_and_uses_automatic_device_selection():
    module = load_main_module()

    args = module.parse_args(["--models", "TCN"])
    assert args.models == ["tcn"]
    selected = module.select_model_registry(["TCN"])
    assert [factory.model_name for factory, _ in selected] == ["TCN"]
    assert selected[0][1]["device"] == [None]


def test_cli_parses_multiple_models():
    module = load_main_module()

    args = module.parse_args(["--models", "mlp", "gru", "xgboost"])

    assert args.models == ["mlp", "gru", "xgboost"]


def test_standard_defaults_include_graph_features_and_all_horizons():
    module = load_main_module()
    from pipeline.experiment_pipeline import ExperimentConfig, HOUR_HORIZONS

    for config in (ExperimentConfig(), module.EXPERIMENT):
        assert config.forecast_horizons == HOUR_HORIZONS
        assert config.graph_embedding.adjacency_mode == "nem"


def test_cli_rejects_removed_configuration_switches():
    import pytest
    module = load_main_module()
    for arguments in (["--graph-embedding", "none"], ["--forecast-mode", "single"],
                      ["--horizon-minutes", "30"], ["--sweep"]):
        with pytest.raises(SystemExit):
            module.parse_args(arguments)
    assert module.parse_args(["--shap-horizons", "5", "60"]).shap_horizons == [5, 60]


def test_cli_passes_standard_defaults_to_the_runner(monkeypatch):
    module = load_main_module()
    calls = []
    monkeypatch.setattr(module, "main", lambda models, **kwargs: calls.append((models, kwargs)))
    module.run_cli(["--models", "tcn", "--device", "xpu", "--plot-seed", "7"])
    models, options = calls[0]
    assert models == ["tcn"]
    assert options["config"].forecast_horizons == tuple(range(5, 61, 5))
    assert options["config"].graph_embedding.adjacency_mode == "nem"
    assert options["device"] == "xpu"
    assert options["plot_seed"] == 7
    assert options["config"].random_seed == module.EXPERIMENT.random_seed
    assert "graph_embeddings" not in options


def run_fake_models(module, monkeypatch, **arguments):
    """Run ``main`` with stub models and report what each one was given."""

    def fake_model(name):
        class Fake:
            model_name = name

            def __init__(self, depth):
                self.depth = depth

            def get_model_name(self):
                return name

        return Fake

    first, second, third = (fake_model(name) for name in ("First", "Second", "Third"))
    nem = module.EXPERIMENT.graph_embedding
    prepared_configs, searches = [], []

    def fake_prepare(config, **kwargs):
        prepared_configs.append(config.graph_embedding)
        rows = 2
        test_df = pd.DataFrame(
            {
                "settlement_date": pd.date_range("2025-01-01", periods=rows, freq="5min"),
                **{f"target_rrp_h{h:02d}": 1.0 for h in module.HOUR_HORIZONS},
            }
        )
        return SimpleNamespace(prepared=SimpleNamespace(test_df=test_df), test_df=test_df, config=config)

    def fake_search(factory, grid, experiment):
        searches.append((factory.model_name, experiment.config.graph_embedding, grid))
        return SimpleNamespace(best_config={"depth": 1})

    registry = [(first, {"depth": [1]}), (second, {"depth": [1, 2]}), (third, {"depth": [1]})]
    monkeypatch.setattr(module, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(module, "prepare_experiment", fake_prepare)
    monkeypatch.setattr(module, "print_experiment_summary", lambda value: None)
    monkeypatch.setattr(module, "search_hyperparameters", fake_search)
    monkeypatch.setattr(module, "final_training_data", lambda model, prepared: (None, None))
    monkeypatch.setattr(first, "train", lambda self, x, y: None, raising=False)
    monkeypatch.setattr(second, "train", lambda self, x, y: None, raising=False)
    monkeypatch.setattr(third, "train", lambda self, x, y: None, raising=False)
    monkeypatch.setattr(module, "predict_test", lambda model, prepared: np.zeros(2))
    monkeypatch.setattr(module, "reconstruct_price_forecast", lambda frame, y, config: y)
    monkeypatch.setattr(
        module, "evaluate", lambda model, actual, predicted: model.get_model_name()
    )
    monkeypatch.setattr(module, "print_evaluation_summary", lambda *args: None)

    module.main(save_results=False, **arguments)
    return nem, prepared_configs, searches


def test_models_sharing_an_embedding_share_one_prepared_experiment(monkeypatch):
    module = load_main_module()

    nem, prepared, searches = run_fake_models(module, monkeypatch)

    assert prepared == [nem]
    assert [(name, embedding) for name, embedding, _ in searches] == [
        ("First", nem),
        ("Second", nem),
        ("Third", nem),
    ]


def test_graph_features_can_be_disabled_only_through_experiment_config(monkeypatch):
    from dataclasses import replace

    module = load_main_module()
    _, prepared, searches = run_fake_models(
        module, monkeypatch,
        config=replace(module.EXPERIMENT, graph_embedding=None), device="xpu",
    )
    assert prepared == [None]
    assert all(embedding is None for _, embedding, _ in searches)


def test_main_passes_each_registry_grid_to_tuning(monkeypatch):
    module = load_main_module()

    _, _, searches = run_fake_models(module, monkeypatch)
    grids = {name: grid for name, _, grid in searches}

    assert grids == {
        "First": {"depth": [1]},
        "Second": {"depth": [1, 2]},
        "Third": {"depth": [1]},
    }


def test_main_uses_prepared_features_target_and_reconstructed_prices(monkeypatch):
    module = load_main_module()

    train_x = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    train_y = pd.DataFrame({f"target_h{h:02d}": [0.5, -0.5] for h in module.HOUR_HORIZONS})
    test_x = pd.DataFrame({"feature_a": [5.0, 6.0], "feature_b": [7.0, 8.0]})
    test_df = pd.DataFrame(
        {
            "rrp": [100.0, 200.0],
            **{f"target_rrp_h{h:02d}": [110.0, 190.0] for h in module.HOUR_HORIZONS},
            "target": [10.0, -10.0],
        }
    )
    prepared = SimpleNamespace(
        forecast_horizons=module.HOUR_HORIZONS,
        embedding_columns=["feature_b"],
        X_train=train_x,
        y_train=train_y,
        X_test=test_x,
        test_df=test_df,
        features=["feature_a", "feature_b"],
    )
    experiment = SimpleNamespace(
        prepared=prepared,
        test_df=test_df,
        features=prepared.features,
        config=module.EXPERIMENT,
    )

    trained = {}

    class FakeModel:
        model_name = "Fake model"

        def __init__(self, depth):
            trained["depth"] = depth

        def train(self, x_train, y_train):
            trained["x"] = x_train.copy()
            trained["y"] = y_train.copy()

        def predict(self, x_test):
            pd.testing.assert_frame_equal(x_test, test_x)
            return np.tile([[5.0], [-5.0]], (1, 12))

        def get_model_name(self):
            return self.model_name

    def fake_search(model_factory, grid, supplied_experiment):
        assert model_factory is FakeModel
        assert grid == {"depth": [1]}
        assert supplied_experiment is experiment
        return SimpleNamespace(best_config={"depth": 1})

    def fake_reconstruct(frame, predicted_target, config):
        assert frame is test_df
        assert config is module.EXPERIMENT
        np.testing.assert_array_equal(predicted_target, np.tile([[5.0], [-5.0]], (1, 12)))
        return frame["rrp"].to_numpy()[:, None] + predicted_target

    evaluated = {}

    def fake_evaluate(model, actual, predicted):
        evaluated["actual"] = actual.copy()
        evaluated["predicted"] = predicted.copy()
        return SimpleNamespace(model_name=model.get_model_name())

    monkeypatch.setattr(module, "MODEL_REGISTRY", [(FakeModel, {"depth": [1]})])
    monkeypatch.setattr(module, "prepare_experiment", lambda config, **kwargs: experiment)
    monkeypatch.setattr(module, "print_experiment_summary", lambda value: None)
    monkeypatch.setattr(module, "search_hyperparameters", fake_search)
    monkeypatch.setattr(module, "reconstruct_price_forecast", fake_reconstruct)
    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(module, "print_evaluation_summary", lambda *args: None)

    module.main(save_results=False)

    pd.testing.assert_frame_equal(trained["x"], train_x)
    pd.testing.assert_frame_equal(trained["y"], train_y)
    assert trained["depth"] == 1
    np.testing.assert_array_equal(evaluated["actual"], np.tile([[110.0], [190.0]], (1, 12)))
    np.testing.assert_array_equal(evaluated["predicted"], np.tile([[105.0], [195.0]], (1, 12)))
