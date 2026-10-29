"""Trainable LSTM model assembled from a PyTorch layer and the shared pipeline."""

from __future__ import annotations

from torch import nn

from electricity_price_forecasting_model.final_model.gru import GRUModel
from electricity_price_forecasting_model.layers.lstm import LSTMNetwork


class LSTMModel(GRUModel):
    """Forecast with an LSTM using the same sequence training policy as GRU."""

    model_name = "LSTM"

    def _build_network(self, input_size: int) -> nn.Module:
        return LSTMNetwork(
            input_size, self.hidden_size, self.num_layers, output_size=self.output_size, dropout=self.dropout
        )
