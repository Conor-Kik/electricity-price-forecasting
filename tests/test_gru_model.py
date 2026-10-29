# ruff: noqa: E402
# Check PyTorch availability before importing dependent modules.
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from electricity_price_forecasting_model.final_model.gru import (
    GRUModel,
    GRUNetwork,
)
from pipeline.sequences import ContextSequenceDataset, SequenceDataset


def test_gru_receives_named_hyperparameters():
    model = GRUModel(
        sequence_length=6,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=1e-5,
        batch_size=4,
        max_epochs=2,
        patience=2,
        min_delta=0.01,
        loss="mae",
        device="cpu",
    )

    assert model.sequence_length == 6
    assert model.hidden_size == 16
    assert model.num_layers == 2
    assert model.dropout == 0.1
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
def test_gru_builds_selected_loss(loss, criterion_type):
    model = GRUModel(loss=loss, device="cpu")

    assert isinstance(model._build_criterion(), criterion_type)


def test_gru_network_matches_original_architecture():
    network = GRUNetwork(input_size=5, hidden_size=16, num_layers=2, dropout=0.2)

    assert network.gru.input_size == 5
    assert network.gru.hidden_size == 16
    assert network.gru.num_layers == 2
    assert network.gru.dropout == pytest.approx(0.2)
    assert network.output.in_features == 16
    assert network.output.out_features == 1
    assert network(torch.zeros(3, 6, 5)).shape == (3, 1)


def test_sequence_datasets_align_targets_and_use_history_context():
    inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    targets = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
    training = SequenceDataset(inputs, targets, sequence_length=3)

    first_window, first_target = training[0]
    assert torch.equal(first_window, inputs[:3])
    assert torch.equal(first_target, targets[2])

    future = torch.arange(10, 16, dtype=torch.float32).reshape(3, 2)
    prediction = ContextSequenceDataset(inputs, future, sequence_length=3)
    (first_prediction_window,) = prediction[0]
    (last_prediction_window,) = prediction[2]
    assert torch.equal(first_prediction_window, torch.cat((inputs[-2:], future[:1])))
    assert torch.equal(last_prediction_window, future)


def test_gru_trains_and_predicts_every_row_with_training_context():
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
    model = GRUModel(
        sequence_length=3,
        hidden_size=4,
        batch_size=2,
        max_epochs=2,
        random_seed=7,
        device="cpu",
    )

    model.train(x_train, y_train)
    predictions = model.predict(x_test)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert len(model.training_loss_history) == 2
    assert model.training_context is not None
    assert len(model.training_context) == 2


def test_gru_early_stops_and_restores_best_validation_epoch(monkeypatch):
    x_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0]})
    y_train = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    x_validation = pd.DataFrame({"feature": [5.0, 6.0]})
    y_validation = pd.Series([5.0, 6.0])
    validation_scores = iter([5.0, 4.0, 4.0, 4.0, 3.0])
    model = GRUModel(
        sequence_length=2,
        hidden_size=4,
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
