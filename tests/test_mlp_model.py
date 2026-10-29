from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("torch")
from torch import nn  # noqa: E402

from electricity_price_forecasting_model.final_model.mlp import (  # noqa: E402
    FeedForwardNetwork,
    MLPModel,
)


def test_mlp_receives_named_hyperparameters():
    model = MLPModel(
        hidden_sizes=(16, 8),
        dropout=(0.1, 0.05),
        learning_rate=3e-4,
        weight_decay=1e-5,
        batch_size=4,
        max_epochs=2,
        patience=2,
        min_delta=0.01,
        loss="mae",
        device="cpu",
    )

    assert model.hidden_sizes == (16, 8)
    assert model.dropout == (0.1, 0.05)
    assert model.learning_rate == 3e-4
    assert model.weight_decay == 1e-5
    assert model.batch_size == 4
    assert model.max_epochs == 2
    assert model.patience == 2
    assert model.min_delta == 0.01
    assert model.loss == "mae"


@pytest.mark.parametrize(
    ("loss", "criterion_type"),
    [
        ("huber", nn.HuberLoss),
        ("mse", nn.MSELoss),
        ("mae", nn.L1Loss),
    ],
)
def test_mlp_builds_selected_loss(loss, criterion_type):
    model = MLPModel(loss=loss, device="cpu")

    assert isinstance(model._build_criterion(), criterion_type)


def test_mlp_rejects_unknown_loss():
    with pytest.raises(ValueError, match="loss must be"):
        MLPModel(loss="unknown", device="cpu")


def test_feed_forward_network_uses_requested_hidden_sizes():
    network = FeedForwardNetwork(3, (16, 8), dropout=0.1)
    linear_layers = [
        layer for layer in network.network if layer.__class__.__name__ == "Linear"
    ]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (3, 16),
        (16, 8),
        (8, 1),
    ]


def test_default_network_matches_architecture():
    model = MLPModel(device="cpu")
    network = FeedForwardNetwork(10, model.hidden_sizes, model.dropout)

    assert model.hidden_sizes == (128, 64, 32)
    assert model.dropout == (0.2, 0.15, 0.0)
    assert [type(layer).__name__ for layer in network.network] == [
        "Linear",
        "LayerNorm",
        "GELU",
        "Dropout",
        "Linear",
        "LayerNorm",
        "GELU",
        "Dropout",
        "Linear",
        "GELU",
        "Linear",
    ]


def test_mlp_trains_and_predicts_all_rows_with_missing_features():
    x_train = pd.DataFrame(
        {
            "price_lag": [10.0, 12.0, np.nan, 16.0, 18.0, 20.0],
            "demand": [100.0, 105.0, 110.0, np.nan, 120.0, 125.0],
        }
    )
    y_train = pd.Series([11.0, 13.0, 15.0, 17.0, 19.0, 21.0], name="target")
    x_test = pd.DataFrame(
        {
            "price_lag": [14.0, np.nan],
            "demand": [108.0, 130.0],
        }
    )
    model = MLPModel(
        hidden_sizes=(8,),
        dropout=0.0,
        batch_size=3,
        max_epochs=2,
        random_seed=7,
        device="cpu",
    )

    model.train(x_train, y_train)
    predictions = model.predict(x_test)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert len(model.training_loss_history) == 2


def test_mlp_early_stops_and_restores_best_validation_epoch(monkeypatch):
    x_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0])
    x_validation = pd.DataFrame({"feature": [4.0, 5.0]})
    y_validation = pd.Series([4.0, 5.0])
    validation_scores = iter([5.0, 4.0, 4.0, 4.0, 3.0])
    model = MLPModel(
        hidden_sizes=(4,),
        dropout=0.0,
        batch_size=2,
        max_epochs=10,
        patience=2,
        min_delta=0.001,
        device="cpu",
    )
    monkeypatch.setattr(
        model,
        "_validation_mae",
        lambda *args: next(validation_scores),
    )

    result = model.train_with_validation(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    assert model.best_epoch == 2
    assert model.stopped_epoch == 4
    assert len(model.training_loss_history) == 4
    assert model.validation_mae_history == [5.0, 4.0, 4.0, 4.0]
    assert result is model.training_result
    assert result.best_epoch == 2
