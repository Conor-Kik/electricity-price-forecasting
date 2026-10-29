# Model implementations

[Project README](../README.md) · [Development guide](../docs/development.md)

This package contains the models used by `main.py`. Training runs are
configured in the root script and coordinated by `pipeline/`.

## Layout

| Directory | Role |
| --- | --- |
| `final_model/` | MLP, GRU, LSTM, TCN and Transformer wrappers |
| `baseline_models/` | XGBoost and ARIMA wrappers |
| `layers/` | PyTorch networks and component layers |
| `layers/gnn_layers/` | Cross-region graph encoder used during feature preparation |

Configure experiments through `EXPERIMENT` in [`main.py`](../main.py).

## Wrappers and networks

A network is a `torch.nn.Module` that maps a batch of features or observed
windows to forecasts.

A wrapper exposes the common `Model` interface. Neural wrappers inherit
[`TorchForecastModel`](../pipeline/neural.py), which supplies fitted
preprocessing, target transforms, training, early stopping and inference.
Network construction belongs in `_build_network(input_size)`.

The main experiment passes twelve target columns. Construct neural output
heads with `self.output_size`; row networks receive `(batch, features)` and
sequence networks receive `(batch, lookback, features)`. Both return
`(batch, self.output_size)`. Public predictions have shape `(N, 12)` and
remain on the experiment's target scale until price reconstruction.

XGBoost shares one horizon-conditioned ensemble across the twelve leads.
ARIMA forecasts the leads from one fitted price process. The graph encoder
adds `gnn_emb_*` inputs; it is separate from the downstream forecasters.

## Add or run a model

Follow the [model-development steps](../docs/development.md#add-a-model), then
register the class and parameter grid in `main.py`. Use the shared experiment
data rather than adding a separate ingestion or evaluation loop.

From the repository root:

```bash
python main.py --models mlp --epochs 1 --skip-shap
python -m pytest -q tests/test_mlp_model.py tests/test_model_pipeline.py
```

The experiment command can build data and GNN caches on its first run;
`--epochs` limits only the deep forecaster.
