"""Forecast reliability features: the recent accuracy of AEMO's price forecast."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pipeline.experiment_pipeline as experiment_pipeline
from pipeline.experiment_pipeline import (
    ExperimentConfig,
    prepare_experiment,
    with_forecast_reliability,
)
from pipeline.preprocessing import (
    FeatureConfig,
    add_forecast_reliability_features,
    build_features,
    forecast_reliability_columns,
    prepare_model_data,
)

DAY = 288  # 5-minute intervals
SMALL = FeatureConfig(
    price_lags=(1,),
    demand_lags=(1,),
    delta_intervals=(1,),
    rolling_windows=(2,),
    p5min_horizons_minutes=(5, 30),
)


def aligned_frame(price, *, region="VIC1", bias_5=0.0, bias_30=0.0) -> pd.DataFrame:
    """Prices beside AEMO forecasts that equal the realised price plus a bias.

    The forecast issued at t for t + 5 minutes is ``p5_h05_rrp`` at t, so it is
    the price one row later; the 30-minute forecast is the price six rows later.
    """
    price = pd.Series(np.asarray(price, dtype=float))
    return pd.DataFrame(
        {
            "settlement_date": pd.date_range(
                "2025-01-01 00:05", periods=len(price), freq="5min"
            ),
            "region": region,
            "rrp": price,
            "p5_h05_rrp": price.shift(-1) + bias_5,
            "p5_h30_rrp": price.shift(-6) + bias_30,
        }
    )


def raw_market(rows=700, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw actuals and raw P5MIN runs, as the ingestion layer supplies them."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01 00:05", periods=rows, freq="5min")
    price = 80 + rng.normal(0, 25, rows)
    price[[150, 151, 520]] = [450.0, 900.0, 600.0]
    actuals = pd.DataFrame(
        {
            "settlement_date": times,
            "region": "VIC1",
            "rrp": price,
            "totaldemand": 5000 + rng.normal(0, 100, rows),
            "availablegeneration": 7000.0,
        }
    )
    runs = [
        pd.DataFrame(
            {
                "REGIONID": "VIC1",
                # A run is labelled one interval after the time it was available.
                "RUN_DATETIME": times + pd.to_timedelta(5, unit="m"),
                "INTERVAL_DATETIME": times + pd.to_timedelta(minutes, unit="m"),
                "RRP": 80 + rng.normal(0, 60, rows),
            }
        )
        for minutes in (5, 30)
    ]
    return actuals, pd.concat(runs, ignore_index=True)


def prepared_market(split=400):
    actuals, p5min = raw_market()
    boundary = actuals["settlement_date"].iloc[split]
    early = p5min["RUN_DATETIME"] <= boundary
    return prepare_model_data(
        region="VIC1",
        horizon=5,
        config=SMALL,
        train_data=SimpleNamespace(df=actuals.iloc[:split], p5min=p5min[early]),
        test_data=SimpleNamespace(df=actuals.iloc[split:], p5min=p5min[~early]),
    )


def test_builds_nine_named_features_for_each_forecast_horizon():
    frame = aligned_frame(np.full(DAY, 100.0))

    features = add_forecast_reliability_features(frame, FeatureConfig())

    columns = forecast_reliability_columns(features)
    assert len(columns) == 18
    assert columns[:9] == [
        "reliability_rrp_err05_mean_12",
        "reliability_rrp_err05_abs_12",
        "reliability_rrp_err05_mean_72",
        "reliability_rrp_err05_abs_72",
        "reliability_rrp_err05_mean_288",
        "reliability_rrp_err05_abs_288",
        "reliability_warnings05_288",
        "reliability_false_alarms05_288",
        "reliability_missed05_288",
    ]
    assert columns[9:] == [name.replace("05", "30") for name in columns[:9]]
    assert list(features["settlement_date"]) == list(frame["settlement_date"])


def test_errors_match_a_known_forecast_bias():
    # AEMO always forecasts $10 too high at 5 minutes and $5 too low at 30.
    frame = aligned_frame(np.linspace(50, 150, DAY), bias_5=10.0, bias_30=-5.0)

    row = add_forecast_reliability_features(frame, FeatureConfig()).iloc[100]

    assert row["reliability_rrp_err05_mean_12"] == pytest.approx(-10.0)
    assert row["reliability_rrp_err05_abs_72"] == pytest.approx(10.0)
    assert row["reliability_rrp_err30_mean_12"] == pytest.approx(5.0)
    assert row["reliability_rrp_err30_abs_288"] == pytest.approx(5.0)


def test_counts_warnings_false_alarms_and_missed_spikes():
    price = np.full(DAY, 100.0)
    price[150] = 400.0  # a spike
    frame = aligned_frame(price)
    frame.loc[99, "p5_h05_rrp"] = 500.0  # warns of row 100, which stays at $100
    frame.loc[149, "p5_h05_rrp"] = 100.0  # no warning of row 150, which is $400

    last = add_forecast_reliability_features(frame, FeatureConfig()).iloc[-1]

    assert last["reliability_warnings05_288"] == 1
    assert last["reliability_false_alarms05_288"] == 1
    assert last["reliability_missed05_288"] == 1


def test_values_never_depend_on_later_data():
    price = 100 + np.random.default_rng(0).normal(0, 30, 3 * DAY)
    frame = aligned_frame(price, bias_5=3.0)
    cut = 2 * DAY

    full = add_forecast_reliability_features(frame, FeatureConfig())
    past_only = add_forecast_reliability_features(frame.iloc[:cut], FeatureConfig())

    columns = forecast_reliability_columns(full)
    pd.testing.assert_frame_equal(past_only[columns], full.iloc[:cut][columns])


def test_regions_are_scored_separately():
    price = 100 + np.random.default_rng(1).normal(0, 30, DAY)
    victoria = aligned_frame(price, region="VIC1", bias_5=10.0)
    tasmania = aligned_frame(price[::-1], region="TAS1", bias_5=-4.0)

    together = add_forecast_reliability_features(
        pd.concat([victoria, tasmania], ignore_index=True), FeatureConfig()
    )

    for frame in (victoria, tasmania):
        alone = add_forecast_reliability_features(frame, FeatureConfig())
        rows = together[together["region"] == frame["region"].iloc[0]]
        columns = forecast_reliability_columns(alone)
        pd.testing.assert_frame_equal(
            rows[columns].reset_index(drop=True), alone[columns]
        )


def test_a_missing_interval_does_not_shift_the_comparison():
    config = FeatureConfig(reliability_horizons_minutes=(5,), reliability_windows=(1,))
    frame = aligned_frame(np.linspace(50, 150, 40), bias_5=10.0).drop(index=20)

    features = add_forecast_reliability_features(frame, config)

    error = features.set_index("settlement_date")["reliability_rrp_err05_mean_1"]
    after_gap = frame["settlement_date"].loc[21]
    # Nothing was issued five minutes before this row, so it has no error.
    assert np.isnan(error.loc[after_gap])
    assert np.allclose(error.drop(index=[after_gap, error.index[0]]), -10.0)


def test_frames_without_aemo_forecasts_get_no_reliability_features():
    frame = aligned_frame(np.full(20, 100.0)).drop(
        columns=["p5_h05_rrp", "p5_h30_rrp"]
    )

    features = add_forecast_reliability_features(frame, FeatureConfig())

    assert forecast_reliability_columns(features) == []
    pd.testing.assert_frame_equal(features, frame)


def test_test_rows_match_one_continuous_build():
    actuals, p5min = raw_market()
    split = 400
    early = p5min["RUN_DATETIME"] <= actuals["settlement_date"].iloc[split]

    whole = build_features(actuals, p5min=p5min, config=SMALL)
    test = build_features(
        actuals.iloc[split:],
        p5min=p5min[~early],
        p5min_history=p5min[early],
        history=actuals.iloc[:split],
        config=SMALL,
    )

    columns = forecast_reliability_columns(whole)
    assert len(columns) == 18
    # The day-long window of the first test row reaches back 288 rows, and its
    # oldest error needs the forecast issued six rows before that.
    assert SMALL.max_history_intervals == 294
    pd.testing.assert_frame_equal(
        test[columns], whole.iloc[split:][columns].reset_index(drop=True)
    )


def test_reliability_features_are_prepared_but_are_not_baseline_inputs():
    prepared = prepared_market()

    built = forecast_reliability_columns(prepared.train_df)
    assert len(built) == 18
    assert set(built) <= set(prepared.test_df.columns)
    assert not set(built) & set(prepared.features)
    assert not set(built) & set(prepared.X_train.columns)


def test_switch_appends_the_reliability_features_to_the_model_inputs():
    prepared = prepared_market()
    baseline = list(prepared.features)

    switched = with_forecast_reliability(prepared)

    extra = forecast_reliability_columns(prepared.train_df)
    assert switched.features == [*baseline, *extra]
    assert list(switched.X_train.columns) == switched.features
    assert list(switched.X_test.columns) == switched.features
    pd.testing.assert_series_equal(switched.y_train, prepared.y_train)
    assert prepared.features == baseline  # the original is left as it was


def test_switch_needs_prepared_reliability_features():
    actuals, _ = raw_market()
    prepared = prepare_model_data(
        region="VIC1",
        horizon=5,
        config=SMALL,
        train_data=SimpleNamespace(df=actuals.iloc[:400]),
        test_data=SimpleNamespace(df=actuals.iloc[400:]),
    )

    with pytest.raises(ValueError, match="no forecast reliability features"):
        with_forecast_reliability(prepared)


@pytest.mark.parametrize("switch", [False, True])
def test_experiment_uses_the_reliability_features_only_when_asked(
    monkeypatch, switch
):
    prepared = prepared_market()
    monkeypatch.setattr(
        experiment_pipeline, "load_prepared_data", lambda config, **_: prepared
    )
    config = ExperimentConfig(
        region="VIC1",
        validation_fraction=0.2,
        forecast_reliability=switch,
    )

    columns = config.graph_embedding.embedding_columns
    for frame in (prepared.train_timeline, prepared.test_timeline, prepared.train_df, prepared.test_df):
        frame[columns] = 0.0
    prepared.features.extend(columns)
    prepared = experiment_pipeline.with_forecast_horizons(prepared, config)
    experiment = prepare_experiment(config)

    extra = forecast_reliability_columns(prepared.train_df)
    assert len(experiment.features) == len(prepared.features) + (18 if switch else 0)
    assert set(extra) <= set(experiment.fit_df.columns)
    assert list(experiment.prepared.X_train.columns) == experiment.features


def test_switch_shares_the_prepared_data_cache():
    off, on = ExperimentConfig(), ExperimentConfig(forecast_reliability=True)

    assert on.cache_file == off.cache_file
    assert on.cache_metadata() == off.cache_metadata()
