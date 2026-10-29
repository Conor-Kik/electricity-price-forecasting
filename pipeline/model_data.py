"""Adapt common experiment timelines to row and sequence estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.contracts import forecast_array, indexed_features, target_columns, target_values


def _forecast_horizons(prepared):
    """Require the complete twelve-horizon contract, including cached data."""
    from pipeline.experiment_pipeline import HOUR_HORIZONS

    if tuple(getattr(prepared, "forecast_horizons", ())) != HOUR_HORIZONS:
        raise ValueError("Forecasting requires all twelve horizons; rebuild the prepared data")
    return HOUR_HORIZONS


@dataclass
class ModelSplit:
    """One fitting/validation fold in the representation a model expects.

    ``score_rows`` selects comparable validation origins. Sequence inputs can
    contain additional unlabeled rows needed as context for later predictions.
    """

    x_train: pd.DataFrame
    y_train: pd.Series | pd.DataFrame
    x_validation: pd.DataFrame
    y_validation: pd.Series | pd.DataFrame
    score_rows: np.ndarray


def selection_data(model, experiment, fit_df=None, validation_df=None) -> ModelSplit:
    """Build a fold without losing observations that are useful only as history.

    Row models receive the eligible supervised rows directly. Sequence models
    receive the full timeline, with ineligible training targets replaced by NaN.
    The optional frames select a particular walk-forward fold.
    """
    fit = experiment.fit_df if fit_df is None else fit_df
    validation = experiment.validation_df if validation_df is None else validation_df
    features = experiment.features
    horizons = _forecast_horizons(experiment.prepared)
    if not getattr(model, "requires_sequence", False):
        return ModelSplit(
            fit[features],
            target_values(fit, horizons),
            validation[features],
            target_values(validation, horizons),
            np.ones(len(validation), dtype=bool),
        )

    timeline = experiment.prepared.train_timeline
    if timeline is None:
        raise ValueError(
            "Sequence models require a complete training timeline; rebuild the cache"
        )
    start = validation["settlement_date"].min()
    end = validation["settlement_date"].max()
    training = timeline.loc[timeline["settlement_date"] < start].copy()
    # The embargo masks labels only. All earlier observations remain context.
    eligible = (
        training["settlement_date"]
        + pd.Timedelta(minutes=experiment.config.horizon_minutes)
        < start
    )
    training.loc[~eligible, target_columns(horizons)] = np.nan
    current = timeline.loc[timeline["settlement_date"].between(start, end)].copy()
    x_train = indexed_features(training, features)
    x_val = indexed_features(current, features)
    y_train, y_val = target_values(training, horizons).copy(), target_values(current, horizons).copy()
    y_train.index, y_val.index = x_train.index, x_val.index
    keys = pd.MultiIndex.from_frame(validation[["region", "settlement_date"]])
    # Match by forecast origin, not by position: the full timeline may have
    # extra rows that were excluded from the supervised validation frame.
    return ModelSplit(x_train, y_train, x_val, y_val, x_val.index.isin(keys))


def final_training_data(model, prepared):
    """Return all pre-test fitting data after hyperparameter selection.

    The sequence timeline includes the unlabeled training tail so the first
    test window can use the most recent observed features.
    """
    horizons = _forecast_horizons(prepared)
    if not getattr(model, "requires_sequence", False):
        return prepared.X_train, prepared.y_train
    if prepared.train_timeline is None:
        raise ValueError("Sequence models require a complete training timeline")
    inputs = indexed_features(prepared.train_timeline, prepared.features)
    targets = target_values(prepared.train_timeline, horizons).copy()
    targets.index = inputs.index
    return inputs, targets


def predict_test(model, prepared):
    """Predict using complete history, then align outputs to scored test rows.

    A prediction at each origin may use observations up to that origin. This
    represents successive rolling forecasts, not one forecast of the entire
    test period issued at the training cutoff.
    """
    outputs = len(_forecast_horizons(prepared))
    if not getattr(model, "requires_sequence", False):
        return forecast_array(model.predict(prepared.X_test), len(prepared.X_test), outputs)
    inputs = indexed_features(prepared.test_timeline, prepared.features)
    history = indexed_features(prepared.train_timeline, prepared.features)
    predicted = forecast_array(model.predict(inputs, history=history), len(inputs), outputs)
    keys = pd.MultiIndex.from_frame(prepared.test_df[["region", "settlement_date"]])
    aligned = pd.DataFrame(predicted, index=inputs.index)
    return aligned.loc[keys].to_numpy()
