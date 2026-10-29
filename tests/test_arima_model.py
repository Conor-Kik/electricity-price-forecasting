from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

from electricity_price_forecasting_model.baseline_models.arima import ARIMAModel  # noqa: E402
from pipeline.experiment_pipeline import (  # noqa: E402
    ExperimentConfig,
    ExperimentData,
    chronological_validation_split,
    reconstruct_price_forecast,
)
from pipeline.model_data import final_training_data, predict_test  # noqa: E402
from pipeline.preprocessing import FeatureConfig, prepare_model_data  # noqa: E402
from pipeline.tuning import search_hyperparameters  # noqa: E402


def price_frame(count=600, seed=0, aemo_bias=None, steps=6):
    """Timestamped AR(1) prices with an optional biased AEMO forecast column."""
    rng = np.random.default_rng(seed)
    price = np.empty(count)
    price[0] = 80.0
    for t in range(1, count):
        price[t] = 80.0 + 0.8 * (price[t - 1] - 80.0) + rng.normal(0, 5)
    index = pd.date_range("2024-01-01", periods=count, freq="5min")
    frame = pd.DataFrame({"rrp": price}, index=index)
    if aemo_bias is not None:
        # AEMO's forecast of the interval `steps` ahead, off by a constant.
        future = np.append(price[steps:], np.full(steps, np.nan))
        frame[f"p5_h{steps * 5:02d}_rrp"] = future - aemo_bias
    return frame


def change_target(frame, steps=6):
    return frame["rrp"].shift(-steps) - frame["rrp"]


def plain_arima(**kwargs):
    settings = {
        "p": 1,
        "d": 0,
        "q": 0,
        "daily_harmonics": 0,
        "fit_window_days": None,
        "transform": "none",
    }
    return ARIMAModel(**{**settings, **kwargs})


def test_arima_forecasts_match_statsmodels_at_each_origin():
    frame = price_frame()
    train, test = frame.iloc[:500], frame.iloc[500:]
    model = plain_arima(p=2, q=1, target_type="price")
    model.train(train, frame["rrp"].shift(-6).iloc[:500])

    predicted = model.predict(test, history=train)

    mean = model.seasonal_coefficients[0]
    residual = frame["rrp"].to_numpy() - mean
    for origin in (500, 530, 593):
        arima = SARIMAX(
            residual[: origin + 1], order=(2, 0, 1), trend="n", concentrate_scale=True
        )
        expected = arima.filter(model.params).forecast(steps=6)[-1] + mean
        assert predicted[origin - 500] == pytest.approx(expected, abs=1e-8)


def test_arima_random_walk_is_persistence():
    frame = price_frame()
    model = plain_arima(p=0, d=1, q=0)
    model.train(frame.iloc[:500], change_target(frame).iloc[:500])

    predicted = model.predict(frame.iloc[500:], history=frame.iloc[:500])

    np.testing.assert_allclose(predicted, 0.0, atol=1e-8)


def test_arima_target_type_changes_units_not_the_price_forecast():
    frame = price_frame()
    train, test = frame.iloc[:500], frame.iloc[500:]
    change = plain_arima(target_type="change")
    price = plain_arima(target_type="price")
    change.train(train, change_target(frame).iloc[:500])
    price.train(train, frame["rrp"].shift(-6).iloc[:500])

    np.testing.assert_allclose(
        change.predict(test, history=train) + test["rrp"].to_numpy(),
        price.predict(test, history=train),
    )


def test_arima_rejects_targets_from_another_horizon():
    frame = price_frame()
    model = plain_arima(horizon_minutes=5)

    with pytest.raises(ValueError, match="do not match horizon_minutes=5"):
        model.train(frame, change_target(frame, steps=6))


def test_arima_predict_requires_training():
    with pytest.raises(RuntimeError, match=r"train\(\) must be called"):
        plain_arima().predict(price_frame(count=10))


def test_arima_mad_transform_centres_on_the_training_median():
    frame = price_frame()
    model = plain_arima(transform="asinh_mad", fit_window_days=1)
    model.train(frame, change_target(frame))

    recent = frame["rrp"].iloc[-288:].to_numpy()
    assert model.transform_center == pytest.approx(np.median(recent))
    assert model.transform_scale == pytest.approx(
        1.4826 * np.median(np.abs(recent - np.median(recent)))
    )
    values = np.array([-500.0, 0.0, 85.0, 15000.0])
    np.testing.assert_allclose(model._inverse(model._forward(values)), values)


@pytest.mark.parametrize("arguments", [{"p": -1}, {"d": 3}, {"transform": "log"}])
def test_arima_rejects_invalid_settings(arguments):
    with pytest.raises(ValueError):
        ARIMAModel(**arguments)


def test_arima_aemo_error_series_learns_a_constant_forecast_bias():
    frame = price_frame(aemo_bias=12.0)
    train, test = frame.iloc[:500], frame.iloc[500:]
    model = plain_arima(p=0, series="aemo_error")
    model.train(train, change_target(frame).iloc[:500])

    predicted = model.predict(test, history=train)

    # The final rows have no AEMO forecast in this synthetic frame.
    available = test["p5_h30_rrp"].notna().to_numpy()
    actual = change_target(frame).iloc[500:].to_numpy()
    np.testing.assert_allclose(predicted[available], actual[available], atol=1e-6)
    np.testing.assert_allclose(predicted[~available], 0.0)
    assert model.fallback_rows == (~available).sum()


def test_arima_chunked_predictions_match_one_call():
    frame = price_frame(count=400)
    train, test = frame.iloc[:300], frame.iloc[300:]
    model = plain_arima(p=2, d=1, q=1)
    model.train(train, change_target(frame).iloc[:300])

    full = model.predict(test, history=train)
    first = model.predict(test.iloc[:40], history=train)
    second = model.predict(test.iloc[40:], history=frame.iloc[:340])

    np.testing.assert_allclose(np.concatenate([first, second]), full, atol=1e-8)


def test_arima_runs_through_search_and_test_prediction():
    count = 3 * 288
    times = pd.date_range("2024-01-01", periods=count, freq="5min")
    daily = 20 * np.sin(2 * np.pi * np.arange(count) / 288)
    raw = pd.DataFrame(
        {
            "region": "NSW1",
            "settlement_date": times,
            "rrp": 90 + daily + np.random.default_rng(1).normal(0, 3, count),
            "totaldemand": 6000 + 10 * daily,
            "availablegeneration": np.full(count, 9000.0),
        }
    )
    features = ["rrp", "totaldemand"]
    feature_config = FeatureConfig(price_lags=(1,), demand_lags=(1,), rolling_windows=(2,))
    split = int(count * 0.8)
    prepared = prepare_model_data(
        horizon=30,
        features=features,
        config=feature_config,
        train_data=SimpleNamespace(df=raw.iloc[:split]),
        test_data=SimpleNamespace(df=raw.iloc[split:]),
    )
    config = ExperimentConfig(
        validation_fraction=0.25,
        model_features=features,
        feature_config=feature_config,
    )
    from pipeline.experiment_pipeline import with_forecast_horizons
    columns = config.graph_embedding.embedding_columns
    for frame in (prepared.train_timeline, prepared.test_timeline, prepared.train_df, prepared.test_df,
                  prepared.X_train, prepared.X_test):
        frame[columns] = 0.0
    prepared = replace(prepared, features=[*prepared.features, *columns], embedding_columns=columns)
    prepared = with_forecast_horizons(prepared, config)
    fit, validation = chronological_validation_split(prepared.train_df, config)
    experiment = ExperimentData(config, prepared, fit, validation, prepared.test_df)

    search = search_hyperparameters(
        ARIMAModel,
        {"p": [1, 2], "daily_harmonics": [0, 2], "fit_window_days": [None]},
        experiment,
        verbose=False,
    )
    assert search.trials["error"].eq("").all()
    model = ARIMAModel(**search.best_config)
    model.train(*final_training_data(model, prepared))
    predicted = predict_test(model, prepared)
    price = reconstruct_price_forecast(prepared.test_df, predicted, config)

    assert predicted.shape == (len(prepared.test_df), 12)
    assert np.isfinite(price).all()
    assert search.best_config["daily_harmonics"] == 2
