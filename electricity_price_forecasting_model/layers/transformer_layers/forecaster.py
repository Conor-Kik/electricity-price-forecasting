from electricity_price_forecasting_model.layers.transformer_layers.transformerLayer import DecoderTransformer

from torch import nn

class Forecaster(nn.Module):
    def __init__(
            self,
            input_size,
            d_model,
            nhead,
            num_layers,
            dim_feedforward=2048,
            dropout=0.0,
            output_size=1,
            ):
        super(Forecaster, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.transformer = DecoderTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            )
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x, mask=None):
        x = self.input_layer(x)
        x = self.transformer(x, mask)
        return self.output_layer(x[:, -1, :])
