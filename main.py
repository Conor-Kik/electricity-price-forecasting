"""Configure, train and evaluate the shared twelve-horizon experiment."""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from electricity_price_forecasting_model.final_model.transformer import TransformerModel
from electricity_price_forecasting_model.final_model.gru import GRUModel
from electricity_price_forecasting_model.final_model.lstm import LSTMModel
from electricity_price_forecasting_model.final_model.mlp import MLPModel
from electricity_price_forecasting_model.final_model.tcn import TCNModel
from electricity_price_forecasting_model.baseline_models.arima import ARIMAModel
from electricity_price_forecasting_model.baseline_models.xgboost import XGBoostModel
from pipeline.artifacts import create_run_directory, save_model_run, save_model_evaluation, save_run_summary
from pipeline.attribution import ShapSummary, feature_lag_importance, importance_percent
from pipeline.evaluation import ModelResult, compare_models_by_horizon, evaluate, print_evaluation_summary
from pipeline.experiment_pipeline import (
    ExperimentConfig,
    ExperimentData,
    HOUR_HORIZONS,
    prepare_experiment,
    print_experiment_summary,
    reconstruct_price_forecast,
)
from pipeline.model import Model
from pipeline.model_data import final_training_data, predict_test
from pipeline.contracts import indexed_features, price_values
from pipeline.neural import TorchForecastModel, resolve_device
from pipeline.saved_runs import load_saved_run
from pipeline.tuning import SearchResult, search_hyperparameters

EXPERIMENT: ExperimentConfig = ExperimentConfig(
    region="QLD1",
    target_type="change",
    model_features="all",
    validation_fraction=0.20,
)

# Each entry pairs a model constructor with candidate hyperparameter values.
# Search evaluates their Cartesian product; one-element lists fix a setting.
# Keep the search small; candidate counts depend on each model's grid.
# Neural losses compare MAE with Huber at the original learning rate;
# validation early stopping already selects the final epoch budget.
MODEL_REGISTRY: list[tuple[type[Model], dict[str, Sequence]]] = [
    (
        XGBoostModel,
        {
            # Compare a shallower ensemble with the current tree capacity.
            "max_depth": [4, 6],
            "learning_rate": [0.1],
            "min_child_weight": [4],
            "max_estimators": [1000],
        },
    ),
    (
        ARIMAModel,
        {
            # Select differencing on validation data for each region.
            "p": [2],
            "d": [1, 0],
            "q": [0],
            "transform": ["asinh_mad"],
            "daily_harmonics": [0],
            "fit_window_days": [7],
        },
    ),
    (
        MLPModel,
        {
            "hidden_sizes": [(128, 64, 32)],
            "dropout": [(0.2, 0.15, 0.0)],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [256],
            "max_epochs": [30],
            "patience": [3],
            "min_delta": [0.001],
            "loss": ["huber", "mae"],
            "huber_delta": [1.0],
        },
    ),
    (
        GRUModel,
        {
            "sequence_length": [12],
            "hidden_size": [16],
            "num_layers": [1],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [512],
            "prediction_batch_size": [2048],
            "max_epochs": [10],
            "patience": [2],
            "min_delta": [0.001],
            "loss": ["huber", "mae"],
            "huber_delta": [1.0],
        },
    ),
    (
        TransformerModel,
        {
            "sequence_length": [72],
            "d_model": [64],
            "nhead": [4],
            "num_layers": [3],
            "dim_feedforward": [128],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [128],
            "prediction_batch_size": [256],
            "max_epochs": [50],
            "patience": [10],
            "min_delta": [0.001],
            "loss": ["mae"],
            "huber_delta": [1.0],
        },
    ),
    (
        LSTMModel,
        {
            "sequence_length": [12],
            "hidden_size": [64],
            "num_layers": [2],
            "dropout": [0.25],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [512],
            "prediction_batch_size": [2048],
            "max_epochs": [20],
            "patience": [3],
            "min_delta": [0.001],
            "loss": ["huber", "mae"],
            "huber_delta": [1.0],
            "gradient_clip_norm": [1.0],
        },
    ),
    (
        TCNModel,
        {
            "sequence_length": [12],
            "channels": [16],
            "kernel_size": [3],
            "num_blocks": [2],
            "dropout": [0.0],
            "learning_rate": [0.001],
            "weight_decay": [0.0001],
            "batch_size": [512],
            "prediction_batch_size": [2048],
            "max_epochs": [20],
            "patience": [5],
            "min_delta": [0.001],
            "loss": ["mae", "huber"],
            "device": [None],
        },
    )
]


def model_cli_name(model_factory: type[Model]) -> str:
    """Return the stable command-line name for a registered model."""
    return model_factory.model_name.lower().replace(" ", "-")


def select_model_registry(
    selected_models: Sequence[str] | None,
) -> list[tuple[type[Model], dict[str, Sequence]]]:
    """Select registry entries while preserving their configured order."""
    if selected_models is None:
        return MODEL_REGISTRY

    requested = {name.lower() for name in selected_models}
    available = {model_cli_name(factory) for factory, _ in MODEL_REGISTRY}
    unknown = requested.difference(available)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown models: {names}")
    return [
        entry
        for entry in MODEL_REGISTRY
        if model_cli_name(entry[0]) in requested
    ]


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train forecasting models or evaluate a trusted saved run.",
        allow_abbrev=False,
    )
    parser.add_argument("--load-run", type=Path,
                        help="Evaluate trusted saved model bundles using their matching prepared-data cache; never train.")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str.lower,
        choices=sorted(model_cli_name(factory) for factory, _ in MODEL_REGISTRY),
        help="Models to run. Defaults to all registered models.",
    )
    parser.add_argument("--folds", type=int, default=EXPERIMENT.walk_forward_folds)
    parser.add_argument(
        "--metric",
        choices=["mae", "rmse", "sse", "median_ae"],
        default=EXPERIMENT.selection_metric,
    )
    parser.add_argument("--seed", type=int, default=EXPERIMENT.random_seed)
    parser.add_argument("--plot-seed", type=int,
                        help="Seed for selecting forecast examples; defaults to the experiment seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto",
                        help="Neural device. Auto prefers CUDA, then Intel XPU, then CPU.")
    parser.add_argument("--epochs", type=int, help="Override the neural epoch limit in the model grids.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip neural GradientSHAP reporting.")
    parser.add_argument("--shap-horizons", nargs="+", type=int,
                        help="Outputs to explain in minutes; defaults to all forecast horizons.")
    parser.add_argument("--shap-top-features", type=int, default=20,
                        help="Number of features in SHAP plots; 0 displays all features.")
    parser.add_argument("--shap-3d", action="store_true",
                        help="Add rotating 3D sequence-importance surfaces with --plot or --save.")
    parser.add_argument("--shap-samples", type=int, default=128)
    parser.add_argument("--shap-draws", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "runs",
    )
    parser.add_argument("--save", action="store_true", help="Save models and result artifacts; disabled by default.")
    parser.add_argument("--plot", action="store_true", help="Display horizon-error plots and a separate SHAP heatmap for each deep model.")
    parser.add_argument("--plot-details", action="store_true", help="Include price-regime plots with --plot or --save.")
    parser.add_argument("--verbose", action="store_true", help="Print full SHAP, timestep, regime and per-horizon tables.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild prepared data from ingestion/cache",
    )
    arguments = list(arguments) if arguments is not None else sys.argv[1:]
    args = parser.parse_args(arguments)
    if args.load_run:
        training_options = {"--refresh", "--epochs", "--folds", "--metric", "--seed"}
        supplied = training_options.intersection(arg.split("=", 1)[0] for arg in arguments)
        if supplied:
            parser.error(f"--load-run restores the saved experiment; cannot use {', '.join(sorted(supplied))}")
    if args.plot_details and not (args.plot or args.save):
        parser.error("--plot-details requires --plot or --save")
    if args.shap_3d and not (args.plot or args.save):
        parser.error("--shap-3d requires --plot or --save")
    if args.shap_top_features < 0:
        parser.error("--shap-top-features must be zero (all features) or positive")
    if args.plot_seed is not None and args.plot_seed < 0:
        parser.error("--plot-seed must be non-negative")
    if min(args.shap_samples, args.shap_draws) < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("Epochs and attribution counts must be positive")
    horizons = HOUR_HORIZONS
    if args.shap_horizons and not set(args.shap_horizons).issubset(horizons):
        parser.error(f"--shap-horizons must select forecast outputs from {horizons}")
    return args


def sequence_attribution_summary(
    lag_attributions: pd.DataFrame, interval_minutes: float,
) -> pd.DataFrame:
    """Summarise contributions by sequence position, not engineered feature lag."""
    lags = sorted({int(column.split("_", 2)[1]) for column in lag_attributions.columns})
    rows = []
    for lag in lags:
        values = lag_attributions.loc[:, lag_attributions.columns.str.startswith(f"lag_{lag}_")]
        rows.append({
            "sequence_lag": lag,
            "minutes_before_prediction": lag * interval_minutes,
            "mean_absolute_contribution": values.abs().sum(axis=1).mean(),
            "mean_signed_contribution": values.sum(axis=1).mean(),
        })
    summary = pd.DataFrame(rows).set_index("sequence_lag")
    summary["absolute_share_percent"] = importance_percent(summary[["mean_absolute_contribution"]]).iloc[:, 0]
    return summary


def report_gradient_shap(model, prepared, reference_features, config, run_directory=None,
                         *, horizons=None, sample_size=128, n_samples=128, verbose=True) -> ShapSummary:
    """Retain small importance summaries for plots; print full diagnostics on request."""
    requested = tuple(sorted(set(horizons or config.horizons)))
    if not set(requested).issubset(config.horizons):
        raise ValueError(f"Attribution horizons must be selected from {config.horizons}")
    history = None
    if model.requires_sequence:
        inputs = indexed_features(prepared.test_timeline, prepared.features)
        history = indexed_features(prepared.train_timeline, prepared.features)
        keys = indexed_features(prepared.test_df, prepared.features).index
        positions = inputs.index.get_indexer(keys)
    else:
        inputs = prepared.X_test
        positions = np.arange(len(inputs))
    importances, feature_lags, residuals = {}, {}, []
    for horizon in requested:
        target = config.horizons.index(horizon)
        if verbose:
            print(f"\nGradientSHAP: +{horizon}min {config.target_type}, "
                  f"samples<={sample_size}, references<=64, draws={n_samples}", flush=True)
        attribution = model.gradient_shap(
            inputs, reference_features, history=history, positions=positions, target=target,
            sample_size=sample_size, baseline_size=64, n_samples=n_samples, device=model.device,
        )
        importance = attribution.attrs["absolute_contributions"].mean().sort_values(ascending=False).rename(
            "mean_absolute_contribution"
        )
        feature_summary = importance.to_frame().join(attribution.mean().rename("mean_signed_contribution"))
        feature_summary["absolute_share_percent"] = importance_percent(importance.to_frame()).iloc[:, 0]
        importances[horizon] = importance
        if model.requires_sequence:
            feature_lags[horizon] = feature_lag_importance(attribution, model.interval_minutes)
        residual = np.abs(attribution.attrs["completeness_residual"])
        residuals.append(float(residual.mean()))
        if verbose:
            print(f"Observations={len(attribution)}, device={attribution.attrs['device']}, "
                  f"reference mean={attribution.attrs['reference_prediction']:.3f} $/MWh")
            print("Top 10 feature contributions (shares use all features and sequence steps):")
            print(feature_summary.head(10).to_string(float_format=lambda value: f"{value:.3f}"))
            print(f"Attribution residual: mean absolute={residual.mean():.4f}, max absolute={residual.max():.4f} $/MWh")
        if model.requires_sequence and (verbose or run_directory is not None):
            timestep_summary = sequence_attribution_summary(attribution.attrs["lag_attributions"], model.interval_minutes)
            if verbose:
                print("Timestep importance (lag 0 = latest observation):")
                print(timestep_summary.to_string(float_format=lambda value: f"{value:.3f}"))
        if run_directory is not None:
            path = run_directory / model_cli_name(type(model))
            suffix = f"_h{horizon:02d}"
            attribution.assign(completeness_residual=attribution.attrs["completeness_residual"]).to_csv(path / f"gradient_shap{suffix}.csv")
            feature_summary.to_csv(path / f"feature_importance{suffix}.csv", index_label="feature")
            if model.requires_sequence:
                attribution.attrs["lag_attributions"].to_csv(path / f"gradient_shap_by_lag{suffix}.csv")
                timestep_summary.to_csv(path / f"gradient_shap_timestep_summary{suffix}.csv")
                feature_lags[horizon].to_csv(path / f"gradient_shap_feature_sequence{suffix}.csv")
    summary = pd.DataFrame(importances).sort_index(axis=1)
    result = ShapSummary(summary, feature_lags)
    relative = result.as_percent()
    if not verbose:
        top = relative.features.mean(axis=1).nlargest(3)
        print("GradientSHAP shares of absolute attribution, averaged over "
              f"{', '.join(map(str, requested))} min: " + "; ".join(f"{name}={value:.1f}%" for name, value in top.items()))
        print(f"Mean absolute attribution residual across explained horizons: {np.mean(residuals):.3f} $/MWh")
    diagnostics = result.diagnostics()
    # Compact output compares the shortest and longest explained targets.
    displayed = diagnostics if verbose else diagnostics.loc[sorted({requested[0], requested[-1]})]
    print("Attribution concentration (shares of total magnitude, not accuracy):")
    print(displayed.to_string(float_format=lambda value: f"{value:.1f}"))
    if run_directory is not None:
        path = run_directory / model_cli_name(type(model))
        summary.to_csv(path / "gradient_shap_features_by_horizon.csv", index_label="feature")
        relative.features.to_csv(path / "gradient_shap_feature_percent_by_horizon.csv", index_label="feature")
        diagnostics.to_csv(path / "gradient_shap_concentration.csv")
        if feature_lags:
            result.lags.to_csv(path / "gradient_shap_sequence_by_horizon.csv", index_label="minutes_before_prediction")
            relative.lags.to_csv(path / "gradient_shap_sequence_percent_by_horizon.csv", index_label="minutes_before_prediction")
    return result


def main(
    selected_models: Sequence[str] | None = None,
    *,
    config=None,
    output_directory=None,
    save_results=False,
    force_refresh=False,
    device=None,
    max_epochs=None,
    skip_shap=False,
    shap_horizons=None,
    shap_samples=128,
    shap_draws=128,
    shap_top_features=20,
    shap_3d=False,
    plot_results=False,
    plot_seed=None,
    plot_details=False,
    verbose=False,
    load_run=None,
) -> None:
    """Train or reload forecasters, then evaluate the held-out test period.

    Temporal representation choices live in model_data; model-specific network
    and fitting details stay behind the common model interface.
    Every model uses twelve targets and the same feature configuration.
    Graph features are enabled unless config.graph_embedding is None.
    Saved-run evaluation requires matching cached inputs and never fits anything.
    """
    if load_run is not None:
        if config is not None or force_refresh or max_epochs is not None:
            raise ValueError("Loading a saved run cannot override its experiment or request training/refresh")
        saved_run = load_saved_run(load_run, selected_models)
        config, prepared = saved_run.config, saved_run.prepared
        entries = [(type(saved.model), {}) for saved in saved_run.models.values()]
        print(f"Evaluating saved models from {saved_run.directory}; tuning and training are skipped.")
    else:
        saved_run = None
        config = config or EXPERIMENT
        if device is not None and config.graph_embedding is not None:
            config = replace(config, graph_embedding=replace(config.graph_embedding, device=device))
        entries = select_model_registry(selected_models)
        experiment: ExperimentData = prepare_experiment(config, force_refresh=force_refresh)
        config, prepared = experiment.config, experiment.prepared
        print_experiment_summary(experiment)
    if shap_horizons and not set(shap_horizons).issubset(config.horizons):
        raise ValueError(f"Attribution horizons must be selected from {config.horizons}")
    run_directory = None
    results: list[ModelResult] = []
    fit_times, shap_importances = {}, {}
    y_test_price = price_values(prepared.test_df, config.horizons)
    if save_results:
        run_directory = create_run_directory(
            config,
            Path(output_directory) if output_directory is not None
            else Path(__file__).resolve().parent / "reports" / "runs",
            prepared,
            source_run=saved_run.directory if saved_run else None,
        )

    for model_factory, hyperparam_grid in entries:
        print(f"=== {model_factory.model_name} ===")
        if saved_run is not None:
            saved_model = saved_run.models[model_cli_name(model_factory)]
            model = saved_model.model
            if isinstance(model, TorchForecastModel) and device is not None:
                model.device = resolve_device(device)
                model.network.to(model.device)
            print(f"Loaded fitted model; prediction device: {getattr(model, 'device', 'cpu')}")
            x_train, _ = final_training_data(model, prepared)
        else:
            hyperparam_grid = dict(hyperparam_grid)
            if issubclass(model_factory, TorchForecastModel):
                if device is not None:
                    hyperparam_grid["device"] = [device]
                if max_epochs is not None:
                    hyperparam_grid["max_epochs"] = [max_epochs]

            fit_started = perf_counter()
            search_result: SearchResult = search_hyperparameters(
                model_factory, hyperparam_grid, experiment
            )

            # Refit from scratch on all pre-test data with the selected epoch budget.
            model: Model = model_factory(**search_result.best_config)
            print(f"Model training/prediction device: {getattr(model, 'device', 'cpu')}")
            x_train, y_train = final_training_data(model, prepared)
            model.train(x_train, y_train)
            fit_times[model.get_model_name()] = perf_counter() - fit_started

        y_pred: np.ndarray = predict_test(model, prepared)
        # Model outputs may be price changes or AEMO residuals. Convert them to
        # actual RRP units before comparing with prices and common baselines.
        predicted_price: np.ndarray = reconstruct_price_forecast(
            prepared.test_df, y_pred, config
        )

        result = evaluate(model, y_test_price, predicted_price)

        results.append(result)
        if run_directory is not None and saved_run is not None:
            save_model_evaluation(run_directory, saved_model, prepared.test_df, predicted_price, result)
        elif run_directory is not None:
            save_model_run(
                run_directory,
                model,
                search_result,
                prepared.test_df,
                predicted_price,
                result,
                extra_columns=(config.aemo_forecast_column,),
                prepared=prepared,
                graph_embedding=config.graph_embedding,
            )
        if isinstance(model, TorchForecastModel) and not skip_shap:
            print(f"Computing GradientSHAP for {model.get_model_name()}...", flush=True)
            shap_importances[model.get_model_name()] = report_gradient_shap(
                model, prepared, x_train, config, run_directory,
                horizons=shap_horizons, sample_size=shap_samples, n_samples=shap_draws, verbose=verbose,
            )

    leaderboard = print_evaluation_summary(results, prepared, verbose, fit_times)
    if run_directory is not None:
        save_run_summary(run_directory, results)
        leaderboard.to_csv(run_directory / "leaderboard.csv")
        compare_models_by_horizon(results, prepared).to_csv(run_directory / "metrics_by_horizon.csv")
        print(f"Run artifacts saved to {run_directory}")
    if plot_results or run_directory is not None:
        # Lazy import keeps text-only runs independent of a GUI backend.
        from pipeline.plotting import present_results

        present_results(results, prepared, shap_importances, show=plot_results,
                        directory=run_directory, details=plot_details,
                        top_features=shap_top_features, sequence_3d=shap_3d,
                        seed=config.random_seed if plot_seed is None else plot_seed)


def run_cli(arguments: Sequence[str] | None = None) -> None:
    """Resolve command-line settings once for both entry points."""
    cli_args = parse_args(arguments)
    main(
        cli_args.models,
        config=replace(
            EXPERIMENT,
            walk_forward_folds=cli_args.folds,
            selection_metric=cli_args.metric,
            random_seed=cli_args.seed,
        ) if cli_args.load_run is None else None,
        load_run=cli_args.load_run,
        output_directory=cli_args.output_dir,
        save_results=cli_args.save,
        force_refresh=cli_args.refresh,
        device=cli_args.device,
        max_epochs=cli_args.epochs,
        skip_shap=cli_args.skip_shap,
        shap_horizons=cli_args.shap_horizons,
        shap_samples=cli_args.shap_samples,
        shap_draws=cli_args.shap_draws,
        shap_top_features=cli_args.shap_top_features,
        shap_3d=cli_args.shap_3d,
        plot_results=cli_args.plot,
        plot_seed=cli_args.plot_seed,
        plot_details=cli_args.plot_details,
        verbose=cli_args.verbose,
    )


if __name__ == "__main__":
    run_cli()
