# Experiment guide

[Back to the README](../README.md)

## Configure a run

`main.py` owns the active `EXPERIMENT` and `MODEL_REGISTRY`.
Change the region or disable GNN features by editing the experiment:

```python
EXPERIMENT = ExperimentConfig(
    region="VIC1",
    target_type="change",
    model_features="all",
    validation_fraction=0.20,
    graph_embedding=None,
)
```

Omit `graph_embedding=None` to use the default NEM graph encoder. To tune it,
import `GraphEmbeddingConfig` from `pipeline.graph_embedding` and pass a
configured instance. GNN configuration applies to every model in a run.

`region` accepts `NSW1` (New South Wales), `QLD1` (Queensland), `VIC1`
(Victoria), `SA1` (South Australia) or `TAS1` (Tasmania). Each run selects
one forecast region; the default GNN encoder uses all five regions as context.

`model_features="all"` selects the standard engineered features. A list of
column names selects specific inputs; enabled graph features are appended.
Future actual prices and target columns must never be model inputs.

Every run forecasts 5, 10, ..., 60 minutes ahead. The target is defined for
each lead, and `main.py` reconstructs prices before evaluation:

| Target | Training label | Price reconstruction |
| --- | --- | --- |
| `change` | Future RRP minus current RRP | Current RRP plus prediction |
| `price` | Future RRP | Prediction |
| `aemo_error` | Future RRP minus aligned P5MIN RRP | P5MIN RRP plus prediction |

## Selection and reproducibility

```bash
# With graph_embedding=None in EXPERIMENT, as in the example above
python main.py --models gru --folds 3 --seed 42 --skip-shap
python main.py --models lstm --skip-shap
```

Each model starts with one configuration, and neural forecasters use MAE loss.
Registry grids use sequences of values; every run evaluates their Cartesian
product. One value fixes a parameter; multiple values compare alternatives.
Edit the grid directly in `MODEL_REGISTRY`, for example setting an LSTM entry's
`"hidden_size": [16, 32, 64]` to compare three sizes.
Start with a small grid when checking a new model.

One fold uses a chronological fit/validation holdout. Additional folds use
expanding training windows and disjoint validation blocks. Labels are
embargoed at the longest forecast lead. Validation errors are pooled across
folds, and the selected model is freshly refitted on all pre-test data.
Neural refits use the median selected epoch across folds.

`--metric` defaults to `mae`; it also accepts `rmse`, `sse` or
`median_ae` and controls neural
checkpoint selection. `--seed` supplies the model seed unless a grid explicitly
sets `random_seed`. The held-out test period is never used for selection.

## Data and caches

Ingestion in `pipeline/data_cleaning/data.py` combines dispatch prices,
regional demand/generation, interconnector conditions and P5MIN forecasts.
Its defaults span all five regions from 1 January 2022 to 31 July 2026, with
the final 20% of the time range reserved for testing. To change ingestion
dates, edit the loader defaults there; these are not CLI options.

Raw caches live under `pipeline/data_cleaning/nemosis_cache/` and
`pipeline/data_cleaning/nemseer_cache/`. Prepared data and graph embeddings
are cached under `model_cache/` by default.

Matching reruns load complete prepared data, including targets and graph
features, but still tune and train the forecasters. `--refresh` rebuilds prepared
data and embeddings; ingestion may reuse downloaded files. It does not force
a fresh download of upstream data.

The graph encoder learns a 60-minute auxiliary target over the NEM graph and
is then frozen. Its fit window must precede every downstream validation block.
If increasing `--folds` causes an overlap error, reduce the encoder's
`fit_fraction` or use fewer folds. Embeddings are reused across target regions.

## Attribution and plots

```bash
python main.py --models mlp gru --plot --shap-horizons 5 30 60
python main.py --models tcn --plot --shap-3d --shap-top-features 30
python main.py --models lstm --save --plot-details --verbose
```

GradientSHAP explains all twelve outputs by default, using up to 128 test
origins, 64 training references and 128 draws per observation. Reduce work
with `--shap-horizons`, `--shap-samples` and `--shap-draws`, or use `--skip-shap`.
Sequence references are complete observed windows.

Attribution plots divide absolute contributions by the total across all
features and sequence steps for each forecast lead, before selecting top
features. These percentages describe attribution shares, not model accuracy.
Raw contributions describe the learned target in $/MWh relative to sampled
reference predictions; the reported residual measures the approximation error.

MLP plots rank features separately for each forecast lead. Sequence plots
separate forecast minutes ahead from observed minutes before the origin and
offer horizon sliders. `--shap-top-features 0` shows all features; `--shap-3d`
adds a rotatable sequence-importance surface. Both `--shap-3d` and
`--plot-details` require `--plot` or `--save`.

Sequence diagnostics describe input-row positions. An engineered input such
as `rrp_lag_288` contains older information even when supplied in the latest
sequence row. Attribution to `gnn_emb_*` describes the embedding feature,
not individual inputs from neighbouring regions.

## Saved artifacts

`--save` creates a unique run directory; `--plot` independently controls
whether plot windows open. Typical saved files are:

| Location | Contents |
| --- | --- |
| Run root | Experiment metadata, leaderboard, summary, per-horizon metrics and comparison plots |
| Each model directory | Trial/fold scores, metadata, `predictions.csv`, `forecast_curves.csv` and `model.joblib` |
| Explained deep model directories | Per-horizon raw SHAP values, feature/sequence summaries, percentages, concentration diagnostics and plots |

Model bundles contain fitted preprocessing and, where needed, sequence
context. Neural bundles are copied to CPU for portability. Graph-enabled
forecasters still require the corresponding embedding inputs: the separate
graph encoder is not bundled. Load only trusted joblib files.

## Re-evaluate a saved run

First save a training run with `--save`. Later, load its run directory:

```bash
python main.py --load-run reports/runs/<run-id> --skip-shap --plot
python main.py --load-run reports/runs/<run-id> --models lstm --device cpu --save
```

Without `--models`, evaluation loads all fitted model bundles in that run.
It uses the saved region, targets, features, GNN settings and seed.
Use `--device` to select the neural models' inference device.

Keep the complete saved-run directory and its matching prepared-data cache.
Changes to preprocessing or raw data can make a saved run incompatible.

This mode evaluates the original held-out test dataset without training.
New raw data would require the separately fitted graph encoder.

Plotting, verbosity and SHAP options remain available. Attribution is still
computed unless `--skip-shap` is supplied. Training options `--refresh`,
`--epochs`, `--folds`, `--metric` and `--seed` are rejected with `--load-run`.
Use `--plot-seed 7` to select different forecast examples. It defaults to the
saved experiment seed and changes only the example selection.

With `--save`, a new directory contains predictions, metrics, plots and any
requested SHAP outputs. The original models remain in the training-run
directory; use that directory for later evaluations.
