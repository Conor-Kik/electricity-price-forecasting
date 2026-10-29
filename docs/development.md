# Development guide

[Back to the README](../README.md)

## Where the code lives

| Module | Responsibility |
| --- | --- |
| `main.py` | Configuration, model registry and run orchestration |
| `pipeline/data_cleaning/` | Raw AEMO ingestion and chronological train/test split |
| `pipeline/preprocessing/` | Feature joins, engineering and scaling utilities |
| `pipeline/graph_embedding/` | Learned cross-region input features |
| `pipeline/experiment_pipeline.py` | Caching, twelve-horizon targets and validation splits |
| `pipeline/model_data.py` | Row/sequence training inputs and test alignment |
| `pipeline/saved_runs.py` | Read-only loading and compatibility checks for saved-run evaluation |
| `pipeline/neural.py`, `pipeline/training.py` | Shared PyTorch lifecycle and checkpoint selection |
| `pipeline/contracts.py`, `pipeline/evaluation.py` | Shape validation, scoring and baseline comparisons |
| `pipeline/attribution.py`, `pipeline/plotting.py`, `pipeline/artifacts.py` | Interpretation, figures and saved runs |
| `electricity_price_forecasting_model/final_model/` | Deep-model wrappers |
| `electricity_price_forecasting_model/baseline_models/` | XGBoost and ARIMA wrappers |
| `electricity_price_forecasting_model/layers/` | Network implementations |

## Data contracts

Raw ingestion returns `Data` with separate `prices`, `region_summary`,
`interconnectors` and `p5min` tables. `Data.df` aliases the raw price table.

`prepare_experiment(config)` returns `ExperimentData` containing the prepared
data, chronological fit/validation frames and held-out test frame.
In the current twelve-horizon experiment:

| Value | Contract |
| --- | --- |
| `prepared.X_train`, `X_test` | DataFrames containing exactly `prepared.features` |
| `prepared.y_train`, `y_test` | DataFrames with twelve ordered `target_h05` through `target_h60` columns |
| `train_df`, `test_df` | Supervised rows with a complete target vector |
| `train_timeline`, `test_timeline` | Full observed feature timelines, including unlabeled context rows |
| `target_rrp_hXX` | Actual future price at the corresponding lead |
| `target_hXX` | Label on the configured target scale at that lead |
| `target`, `target_rrp` | Labels for the longest lead |
| Model predictions | NumPy array of shape `(N, 12)`, ordered by `forecast_horizons` |

Targets are matched by exact region/timestamp. Training vectors are embargoed
at their final outcome. Future observations must never enter the input features.
Lower-level adapters and model wrappers also support scalar targets for
isolated experiments; the main experiment requires all twelve leads.

For feature families, see the [feature reference](../pipeline/preprocessing/feature_list.txt).
The exact inputs for a run are in `prepared.features`. The preprocessing
`example_usage.py` demonstrates the lower-level scalar adapter.

## Add a model

1. Create a wrapper in `final_model/` or `baseline_models/` with a stable
   `model_name` and named constructor parameters for its tunable settings.
2. Implement `Model.train(x_train, y_train)` and `predict(x_test)`, or subclass
   `TorchForecastModel` for a neural architecture. Fit preprocessing only on
   training data and store it for reuse during inference.
3. Support matrix labels and return `(N, 12)` predictions on the configured
   target scale. Return price changes for a `change` experiment; `main.py`
   handles reconstruction into prices. A scalar-only estimator needs an
   explicit multioutput strategy before it can join the shared experiment.
4. Import the class in `main.py` and append `(ModelClass, parameter_grid)` to
   `MODEL_REGISTRY`. Each grid value must be a sequence. The CLI name comes
   from the lowercased model name, with spaces replaced by hyphens.
5. Test shape, feature handling, temporal alignment and prediction behaviour,
   then run the model through the shared selection/refit/evaluation path.

Do not load or split data inside the wrapper. Tuning injects experiment
horizons and target settings into constructors that accept them.

For a neural model, implement `_build_network(input_size)` and construct the
output head with `self.output_size`. Forward passes return
`(batch, self.output_size)`. The shared wrapper supplies imputation, scaling,
target transforms, training, early stopping and batched inference. Existing
MLP and GRU wrappers are working examples.

## Sequence models

Set `requires_sequence = True` and expose `sequence_length`. The network
receives `(batch, sequence_length, features)` inputs. Shared adapters attach
`(region, settlement_date)` as the DataFrame index, outside the numeric inputs.

Validation embargoes mask labels while keeping earlier observed features as
context. Sequence models reject mixed regions, duplicate/reversed timestamps
and missing intervals; they do not interpolate absent observations.

For successive prediction calls, supply explicit recent history:

```python
prediction = model.predict(current_features, history=recent_features)
```

History must end immediately before the current frame and contain enough
rows for the model's lookback. Both frames use the same temporal index.
Prediction does not advance stored history. Without explicit history, the
saved training tail is used and its timestamps are checked.

## Verify a change

```bash
python -m pytest -q
python main.py --models YOUR_MODEL_NAME --skip-shap
```

Tests use synthetic inputs for temporal alignment, leakage checks, model
integration and saved-bundle round trips. The second command runs a real
experiment and may build data/embedding caches. For neural models, add
`--epochs 1` to limit the forecaster while checking integration.
