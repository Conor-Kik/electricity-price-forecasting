# Regression tests

[Project README](../README.md) · [Development guide](../docs/development.md)

Run commands from the repository root, in the environment used for the project:

```bash
python -m pytest -q
```

The default command collects the regression suite in `tests/`.

## Run focused checks

```bash
# Shared pipeline, experiment wiring and hyperparameter selection
python -m pytest -q tests/test_model_pipeline.py tests/test_main.py tests/test_tuning.py

# One architecture
python -m pytest -q tests/test_mlp_model.py

# Graph features and leakage checks
python -m pytest -q tests/test_graph_embedding.py

# Saved-run evaluation
python -m pytest -q tests/test_saved_runs.py

# Stop at the first failure and show a shorter traceback
python -m pytest -x --tb=short
```

## What the suite checks

| Area | Examples |
| --- | --- |
| Model wrappers and networks | Architecture settings, fitting and prediction behaviour |
| Sequence handling | Observed history, timestamp validation, gaps and chunked inference |
| Experiment preparation | Complete targets, embargoes, cache compatibility and graph alignment |
| Model selection | Validation-only scoring, failed trials and final refit settings |
| Evaluation | Shared masks, persistence/P5MIN baselines and price regimes |
| Training and artifacts | Checkpoint restoration, progress display and saved-model round trips |

Tests use small synthetic datasets and temporary output directories.

Neural and graph checks require a working PyTorch installation. An unavailable
GPU does not prevent CPU checks, but an installed PyTorch package with broken
DLL dependencies can fail during collection. Check the pytest summary for
skips as well as failures.

## Add a regression test

Keep fixtures small and deterministic. Verify observable behaviour, especially
target ordering, forecast alignment, training-only preprocessing and inference
context. Use `tmp_path` for saved outputs and mock ingestion when testing data
preparation. Add a regression case for a bug, then run the affected tests and
any shared integration tests it could influence.
