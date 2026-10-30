"""Experiment date ranges reach ingestion and keep cached runs separate."""

from dataclasses import asdict, replace
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipeline.data_cleaning import data as ingestion
from pipeline.experiment_pipeline import ExperimentConfig, load_prepared_data
from pipeline.preprocessing import FeatureConfig
from pipeline.saved_runs import _restore_config


@pytest.mark.parametrize('start,end', [
    ('bad', '2024-01-02'), ('2024-01-01', 'bad'),
    ('2024-01-02', '2024-01-01'), ('2024-01-01', '2024-01-01'),
    ('2024-01-01T00:00:00+10:00', '2024-01-02T00:00:00+10:00'),
])
def test_invalid_date_ranges_are_rejected(start, end):
    with pytest.raises(ValueError):
        ExperimentConfig(start_date=start, end_date=end)


def test_date_ranges_separate_both_caches_and_survive_saved_config_restore():
    config = ExperimentConfig(start_date='2024-01-01', end_date='2024-02-01')
    for other in (replace(config, start_date='2024-01-02'),
                  replace(config, end_date='2024-03-01')):
        assert other.cache_file != config.cache_file
        assert other.graph_embedding_cache_file != config.graph_embedding_cache_file
        assert other.cache_metadata() != config.cache_metadata()
        assert other.graph_embedding_cache_metadata() != config.graph_embedding_cache_metadata()
    assert _restore_config(asdict(config)) == config
    assert replace(config, start_date='2024-01-01T00:00:00').cache_file == config.cache_file


def test_experiment_dates_reach_ingestion_and_matching_cache_avoids_fetch(monkeypatch, tmp_path):
    raw = pd.DataFrame({
        'region': ['NSW1'] * 200,
        'settlement_date': pd.date_range('2024-01-01', periods=200, freq='5min'),
        'rrp': 100 + np.sin(np.arange(200)),
        'totaldemand': 200 + np.arange(200),
        'availablegeneration': np.full(200, 500.),
    })
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(df=raw.iloc[:160]), SimpleNamespace(df=raw.iloc[160:])

    monkeypatch.setattr(ingestion, 'getTrainTestData', fetch)
    config = ExperimentConfig(
        start_date='2024-01-01', end_date='2024-01-02',
        graph_embedding=None, cache_directory=tmp_path,
        model_features=['rrp', 'totaldemand'],
        feature_config=FeatureConfig(price_lags=(1,), demand_lags=(1,), rolling_windows=(2,)),
    )
    first = load_prepared_data(config)
    cached = load_prepared_data(config)
    assert calls == [{'start': datetime(2024, 1, 1), 'end': datetime(2024, 1, 2)}]
    pd.testing.assert_frame_equal(first.X_train, cached.X_train)
    assert first.forecast_horizons == tuple(range(5, 61, 5))
