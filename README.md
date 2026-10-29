# Forecast-Informed Electricity Price Forecasting

Developed for the STAT3007 project at The University of Queensland (UQ).
This project forecasts five-minute Regional Reference Prices
(RRP) in Australia's National Electricity Market. At each forecast origin,
models predict the next hour: 12 prices, from 5 to 60 minutes ahead.

We compare MLP, GRU, LSTM, TCN, Transformer, XGBoost and ARIMA against
persistence (the current price) and available AEMO P5MIN forecasts. Models
share data preparation, chronological validation and held-out test evaluation.

The final project report will be added when complete.

## Setup

Run these commands from the repository root. Python 3.12 or newer and Git must be installed;
Git is needed for the pinned NEMSEER dependency in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU runs need a PyTorch installation compatible with your CUDA or Intel XPU
device. CPU runs are supported.

## Run an experiment

Start with one model and skip attribution to reduce computation:

```bash
python main.py --models mlp --device cpu --epochs 3 --skip-shap
```

The first run can take substantial time and disk space. Default ingestion
covers all five NEM regions from January 2022 to July 2026, and GNN feature
training is enabled. `--epochs` limits the deep forecaster, not the GNN encoder.
Later runs reuse raw data, prepared features and cached embeddings.

Common commands:

```bash
# Compare selected models using automatic device selection
python main.py --models mlp gru xgboost --skip-shap

# Display forecast errors, example price curves and SHAP views
python main.py --models mlp --plot --shap-horizons 5 30 60

# Save results, plots and fitted models
python main.py --models lstm --save --skip-shap

# Show all command-line options
python main.py --help
```

Omitting `--models` runs all seven registered models. Deep models produce
12 outputs, XGBoost shares one horizon-conditioned ensemble across the leads,
and ARIMA projects one fitted price model over the hour.

### Re-evaluate a saved run

```bash
python main.py --load-run reports/runs/<run-id> --skip-shap --plot --plot-seed 7
```

This loads the run's fitted models and original experiment configuration, then
recomputes test predictions and metrics without tuning, training or rebuilding
data. Add `--models lstm` to select a saved model. The matching prepared-data
cache must still exist, including GNN embeddings when enabled. Load only trusted
joblib files.

Change `--plot-seed` to view different example timestamps. Predictions and
metrics stay the same.

`--save` writes evaluation reports into a new run directory and leaves the
original models untouched. See the [experiment guide](docs/experiments.md#re-evaluate-a-saved-run)
for compatible options and limitations.

## Change the configuration

Edit [`main.py`](main.py):

| Setting | Where to change it |
| --- | --- |
| Region, target and input features | `EXPERIMENT` |
| GNN features | `EXPERIMENT.graph_embedding`; set to `None` to disable |
| Model hyperparameters and search grids | `MODEL_REGISTRY` |

The current experiment uses QLD1, price-change targets, all standard
features plus GNN embeddings, and a 20% validation fraction. All experiments
use the complete 12-horizon curve.

Set `EXPERIMENT.region` to one of these NEM region codes:

| Code | State |
| --- | --- |
| `NSW1` | New South Wales |
| `QLD1` | Queensland |
| `VIC1` | Victoria |
| `SA1` | South Australia |
| `TAS1` | Tasmania |

For example, forecast Victoria by changing the configuration in `main.py`:

```python
EXPERIMENT = ExperimentConfig(
    region="VIC1",
    target_type="change",        # "change", "price" or "aemo_error"
    model_features="all",        # Or a list of feature column names
    validation_fraction=0.20,     # Fraction of training rows used for validation
)
```

Each run forecasts one region; enabled GNN features still use the other NEM
regions as context. Add `graph_embedding=None` to disable those features.
Validation fractions must be strictly between 0 and 1.

In `MODEL_REGISTRY`, one value fixes a parameter and multiple values search all
combinations. For example, `"hidden_size": [16, 32, 64]` compares three sizes.

The default grids use one or two candidates per model:

| Model | Parameter searched | Values |
| --- | --- | --- |
| XGBoost | Tree depth | 4, 6 |
| ARIMA | Differencing order (`d`) | 1, 0 |
| MLP, GRU, LSTM, TCN | Training loss | MAE, Huber |
| Transformer | Training loss (fixed) | MAE |

All other settings stay fixed, including neural learning rate (0.001) and each
model's Huber delta. Neural early stopping selects the epoch budget. Training
loss is selected using the configured validation metric (MAE by default).
With one validation fold, each region requires 13 candidate fits and 7 final
refits; additional folds multiply the candidate fits. This is a small comparison
rather than an exhaustive search.

Use `--device cpu|cuda|xpu|auto` to select the device for the GNN encoder and
deep models. Automatic selection prefers CUDA, then XPU, then CPU.

For configuration examples, target definitions, folds and cache behaviour,
see the [experiment guide](docs/experiments.md).

## Results and interpretation

The console reports MAE, RMSE, improvements over baselines and model fit time.
Errors are measured in $/MWh after reconstructing price forecasts.
Add `--verbose` for per-horizon, price-regime and detailed attribution tables.

Hyperparameters and neural checkpoints are selected on chronological validation
data. Training labels are embargoed by the longest horizon (60 minutes), and
the selected model is refitted before evaluation on the held-out test period.
Model comparisons use shared eligible forecast values.

- `--plot` opens forecast and attribution plots.
- `--save` writes to `reports/runs/<run-id>/`; use `--output-dir PATH` to
  choose another parent directory.

Saved runs include comparison tables, per-horizon predictions, forecast curves,
plots and fitted model bundles.

GradientSHAP runs for deep models by default. Its plots show each feature's
share of total absolute attribution, not accuracy or explained variance.
Use `--shap-horizons 5 30 60` to explain fewer outputs, or `--skip-shap` to
skip attribution. See the [experiment guide](docs/experiments.md) for details.

## Tests and development

```bash
python -m pytest -q
```

See the [development guide](docs/development.md) for data shapes, model contracts
and adding an architecture, and the [tests guide](tests/README.md) for
focused verification commands.

| Directory | Contents |
| --- | --- |
| `pipeline/` | Ingestion, features, GNN encoding, training and evaluation |
| `electricity_price_forecasting_model/` | Model wrappers and network layers |
| `tests/` | Regression tests |
| `docs/` | Technical guides |

[Documentation](docs/README.md)

## Authors

The project was developed by:

- [Conor Kikkert](https://github.com/Conor-Kik)
- [Connor McAreavey](https://github.com/ConnorMcareavey)
- [Tyreece Paul](https://github.com/tyreecepaul)
- [John Pioc](https://github.com/johnpioc)
- [Massimo Spadaro](https://github.com/MassiveMOE)
