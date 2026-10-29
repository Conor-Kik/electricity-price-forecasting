"""Comparable model selection with optional expanding-window validation."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.contracts import METRIC_NAMES, forecast_array, score_forecast
from pipeline.experiment_pipeline import ExperimentData, walk_forward_splits
from pipeline.model import Model
from pipeline.model_data import selection_data


@dataclass
class SearchResult:
    """Selected constructor settings plus an audit trail of all candidate runs.

    ``best_config`` already includes ``refit_config`` overrides. ``trials`` has
    one row per candidate; ``folds`` records its individual validation periods.
    """

    best_config: dict
    best_score: float
    metric: str
    trials: pd.DataFrame
    refit_config: dict = field(default_factory=dict)
    folds: pd.DataFrame = field(default_factory=pd.DataFrame)


def search_hyperparameters(
    model_factory: Callable[..., Model],
    hyperparam_grid: dict[str, Sequence],
    experiment: ExperimentData,
    *,
    metric: str | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> SearchResult:
    """Search a parameter grid using only pre-test chronological validation.

    Each candidate starts fresh on every fold, including fitted preprocessing.
    The lowest pooled validation error wins; the returned settings are used
    to train a new model on all available pre-test training data.
    """
    settings = getattr(experiment, "config", None)
    metric = metric or getattr(settings, "selection_metric", "mae")
    seed = getattr(settings, "random_seed", 42) if seed is None else seed
    if metric not in METRIC_NAMES:
        raise ValueError(f"Metric must be one of {METRIC_NAMES}")
    signature = inspect.signature(model_factory).parameters
    # Classical and neural estimators expose different constructor options.
    # Inject shared settings only where the factory accepts them.
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.values()
    )

    def supports(name):
        return name in signature or accepts_kwargs

    candidates = [
        dict(zip(hyperparam_grid, values))
        for values in itertools.product(*hyperparam_grid.values())
    ]
    splits = (
        walk_forward_splits(experiment)
        if settings is not None
        else [(experiment.fit_df, experiment.validation_df)]
    )
    if verbose:
        print(
            f"Searching {len(candidates)} configurations across {len(splits)} fold(s)"
        )

    records, fold_records, effective_configs, refits = [], [], [], []
    for position, candidate in enumerate(candidates, start=1):
        config = candidate.copy()
        if supports("random_seed"):
            config.setdefault("random_seed", seed)
        if supports("selection_metric"):
            # Checkpoint selection and candidate ranking must optimize the same error.
            config["selection_metric"] = metric
        if supports("interval_minutes") and settings is not None:
            config["interval_minutes"] = settings.feature_config.interval_minutes
        if settings is not None:
            # Classical models need explicit horizons; neural heads infer them from y.
            for name in ("horizon_minutes", "target_type", "forecast_horizons"):
                if supports(name):
                    config[name] = getattr(settings, name)
        effective_configs.append(config)
        actuals, predictions, selected_epochs = [], [], []
        refit = {}
        error = ""
        for fold_id, (fit_df, validation_df) in enumerate(splits, start=1):
            try:
                # Never carry fitted weights or scalers from one fold to the next.
                model = model_factory(**config)
                data = selection_data(model, experiment, fit_df, validation_df)
                validation_trainer = getattr(model, "train_with_validation", None)
                training_result = None
                if validation_trainer is None:
                    model.train(data.x_train, data.y_train)
                else:
                    training_result = validation_trainer(
                        data.x_train, data.y_train, data.x_validation, data.y_validation
                    )
                actual = data.y_validation.to_numpy(dtype=float)
                outputs = len(experiment.prepared.forecast_horizons)
                actual = forecast_array(actual, len(data.x_validation), outputs, name="validation targets")
                predicted = forecast_array(model.predict(data.x_validation), len(data.x_validation), outputs)
                finite = np.isfinite(actual)
                rows = data.score_rows & finite.all(axis=1)
                if not rows.any() or not np.isfinite(predicted[rows]).all():
                    raise ValueError(
                        "Model produced missing/non-finite validation results"
                    )
                actual, predicted = actual[rows], predicted[rows]
                actuals.append(actual)
                predictions.append(predicted)
                selected_epoch = getattr(training_result, "best_epoch", None)
                if selected_epoch is not None:
                    selected_epochs.append(int(selected_epoch))
                refit.update(getattr(training_result, "refit_config", {}))
                fold_records.append(
                    {
                        "trial": position,
                        "fold": fold_id,
                        **{
                            key: score_forecast(actual, predicted, key)
                            for key in METRIC_NAMES
                        },
                        "n_rows": len(actual),
                        "selected_epoch": selected_epoch,
                        "error": "",
                        "fit_start": fit_df["settlement_date"].min(),
                        "fit_end": fit_df["settlement_date"].max(),
                        "validation_start": validation_df["settlement_date"].min(),
                        "validation_end": validation_df["settlement_date"].max(),
                    }
                )
            except Exception as exception:  # noqa: BLE001 -- record failed estimator trials and continue search
                error = f"{type(exception).__name__}: {exception}"
                fold_records.append(
                    {"trial": position, "fold": fold_id, "error": error}
                )
                break

        # A final refit has no validation callback. Use the typical selected
        # training duration across folds instead of consulting the held-out test.
        selected_epoch = int(np.median(selected_epochs)) if selected_epochs else None
        if selected_epoch is not None and supports("max_epochs"):
            refit["max_epochs"] = selected_epoch
        refits.append(refit)
        if error:
            metrics = {key: float("nan") for key in METRIC_NAMES}
            n_rows = 0
        else:
            # Pool errors before computing the metric: averaging fold RMSEs
            # would not equal RMSE over all validation observations.
            actual, predicted = np.concatenate(actuals), np.concatenate(predictions)
            metrics = {
                key: score_forecast(actual, predicted, key) for key in METRIC_NAMES
            }
            n_rows = len(actual)
        records.append(
            {
                "trial": position,
                **config,
                **metrics,
                "n_rows": n_rows,
                "selected_epoch": selected_epoch,
                "error": error,
            }
        )
        if verbose:
            outcome = (
                error or f"{metric}={metrics[metric]:.3f} (best epoch {selected_epoch})"
            )
            print(f"[{position}/{len(candidates)}] {outcome}")

    trials = pd.DataFrame.from_records(records)
    if trials.empty:
        raise RuntimeError("No configuration produced a usable score: empty grid")
    scored = trials[np.isfinite(trials[metric])]
    if scored.empty:
        details = "; ".join(trials["error"].astype(str).tolist())
        raise RuntimeError(f"No configuration produced a usable score: {details}")
    best = scored.loc[scored[metric].idxmin()]
    index = int(best["trial"]) - 1
    best_config = {**effective_configs[index], **refits[index]}
    if verbose:
        print(f"Best {metric}: {best[metric]:.3f} with {best_config}")
    return SearchResult(
        best_config,
        float(best[metric]),
        metric,
        trials,
        refit_config=refits[index],
        folds=pd.DataFrame.from_records(fold_records),
    )
