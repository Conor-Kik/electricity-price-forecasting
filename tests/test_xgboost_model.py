from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from electricity_price_forecasting_model.baseline_models.xgboost import XGBoostModel


def test_xgboost_receives_named_hyperparameters():
    model = XGBoostModel(
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=4,
        max_estimators=7,
        early_stopping_rounds=None,
    )

    params = model.model.get_params()

    assert params["objective"] == "reg:absoluteerror"
    assert params["max_depth"] == 3
    assert params["learning_rate"] == 0.1
    assert params["min_child_weight"] == 4
    assert params["n_estimators"] == 7
    assert params["random_state"] == 42


def test_xgboost_can_train_without_a_validation_set():
    model = XGBoostModel(max_estimators=3, early_stopping_rounds=None)
    x_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])

    model.train(x_train, y_train)

    predictions = model.predict(x_train)
    assert predictions.shape == (4,)


def test_xgboost_predict_requires_training():
    model = XGBoostModel(max_estimators=3)

    with pytest.raises(RuntimeError, match=r"train\(\) must be called"):
        model.predict(pd.DataFrame({"feature": [1.0]}))


def test_xgboost_predicts_all_rows_with_missing_features():
    model = XGBoostModel(max_estimators=3)
    x_train = pd.DataFrame(
        {
            "price_lag": [0.0, 1.0, np.nan, 3.0],
            "demand": [10.0, 11.0, 12.0, 13.0],
        }
    )
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])
    x_test = pd.DataFrame(
        {
            "price_lag": [1.5, np.nan],
            "demand": [11.5, 14.0],
        }
    )

    model.train(x_train, y_train)
    predictions = model.predict(x_test)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_xgboost_rejects_internal_early_stopping():
    with pytest.raises(ValueError, match="tune max_estimators"):
        XGBoostModel(early_stopping_rounds=10)
