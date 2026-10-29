from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from electricity_price_forecasting_model.final_model.lstm import LSTMModel  # noqa: E402
from electricity_price_forecasting_model.layers.lstm import LSTMNetwork  # noqa: E402


def test_lstm_network_shape_and_dropout():
    network = LSTMNetwork(input_size=3, hidden_size=8, num_layers=2, dropout=0.2)

    assert network.lstm.batch_first
    assert network.lstm.dropout == pytest.approx(0.2)
    assert network(torch.zeros(4, 5, 3)).shape == (4, 1)
    assert LSTMNetwork(3, num_layers=1, dropout=0.2).lstm.dropout == 0.0


def test_lstm_trains_and_predicts_with_sequence_context():
    x_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y_train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x_test = pd.DataFrame({"feature": [6.0, 7.0]})
    model = LSTMModel(
        sequence_length=3,
        hidden_size=4,
        batch_size=2,
        max_epochs=1,
        device="cpu",
    )

    model.train(x_train, y_train)
    predictions = model.predict(x_test)

    assert isinstance(model.network, LSTMNetwork)
    assert model.get_model_name() == "LSTM"
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert len(model.training_context) == 2
