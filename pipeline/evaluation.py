from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pipeline.contracts import forecast_array, price_values
from pipeline.model import Model
from pipeline.preprocessing import PreparedModelData

METRIC_NAMES: list[str] = ["mae", "rmse", "sse", "median_ae"]
PRICE_REGIMES: tuple[tuple[str, float | None, float | None], ...] = (
    ("Negative (< $0/MWh)", None, 0.0),
    ("Normal ($0-$300/MWh)", 0.0, 300.0),
    ("High ($300-$1,000/MWh)", 300.0, 1000.0),
    ("Extreme (>= $1,000/MWh)", 1000.0, None),
)


@dataclass
class ModelResult:
    model_name: str
    predictions: np.ndarray
    valid_mask: np.ndarray
    metrics: dict[str, float]


def _validate_result_lengths(
    results: list[ModelResult],
    expected_shape: tuple[int, ...],
) -> None:
    for result in results:
        if result.predictions.shape != expected_shape:
            raise ValueError(
                f"{result.model_name} has {result.predictions.size} predictions "
                f"for expected shape {expected_shape}"
            )
        if result.valid_mask.shape != expected_shape:
            raise ValueError(
                f"{result.model_name} has a validity mask of length "
                f"{result.valid_mask.shape} for expected shape {expected_shape}"
            )


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if actual.size == 0:
        return {name: float("nan") for name in METRIC_NAMES}

    errors: np.ndarray = actual - predicted
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "sse": float(np.sum(errors**2)),
        "median_ae": float(np.median(np.abs(errors))),
    }


def evaluate(
    model: Model,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> ModelResult:
    actual = np.asarray(actual, dtype=float)
    outputs = actual.shape[1] if actual.ndim == 2 else 1
    actual = forecast_array(actual, len(actual), outputs, name="actual")
    predicted = forecast_array(predicted, len(actual), outputs)

    valid_mask: np.ndarray = np.isfinite(actual) & np.isfinite(predicted)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if actual.ndim == 2 and supplied.shape == (len(actual),):
            supplied = np.broadcast_to(supplied[:, None], actual.shape)
        valid_mask &= forecast_array(supplied, len(actual), outputs, name="mask").astype(bool)

    return ModelResult(
        model_name=model.get_model_name(),
        predictions=predicted,
        valid_mask=valid_mask,
        metrics=_score(actual[valid_mask], predicted[valid_mask]),
    )


def _comparison_inputs(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]:
    if not results:
        raise ValueError("needs at least one ModelResult")

    test_df = test_data.test_df
    horizons = getattr(test_data, "forecast_horizons", ())
    actual = price_values(test_df, horizons)
    persistence = test_df["rrp"].to_numpy(dtype=float)
    if actual.ndim == 2:
        persistence = np.broadcast_to(persistence[:, None], actual.shape)
    _validate_result_lengths(results, actual.shape)

    shared_mask = np.isfinite(actual) & np.isfinite(persistence)
    for result in results:
        shared_mask &= np.asarray(result.valid_mask, dtype=bool)

    forecasts: list[tuple[str, np.ndarray]] = [
        (result.model_name, np.asarray(result.predictions, dtype=float))
        for result in results
    ]
    forecasts.append(("Persistence", persistence))

    aemo_columns = [f"p5_h{h:02d}_rrp" for h in (horizons or (test_data.horizon,))]
    if all(column in test_df for column in aemo_columns):
        aemo = test_df[aemo_columns].to_numpy(dtype=float) if actual.ndim == 2 else test_df[aemo_columns[0]].to_numpy(dtype=float)
        aemo_mask = np.isfinite(aemo)
        if aemo_mask.any(axis=0).all():
            shared_mask &= aemo_mask
            forecasts.append(("AEMO P5MIN", aemo))

    if not shared_mask.any():
        raise RuntimeError("No test rows are shared by all models and baselines")

    return actual, shared_mask, forecasts


def _comparison_table(
    actual: np.ndarray,
    forecasts: list[tuple[str, np.ndarray]],
    rows: np.ndarray,
) -> pd.DataFrame:
    records = []
    for name, predicted in forecasts:
        records.append(
            {
                "model": name,
                **_score(actual[rows], predicted[rows]),
                "n_rows": int(rows.sum()),
            }
        )

    comparison = pd.DataFrame.from_records(records).set_index("model")
    persistence_mae = comparison.loc["Persistence", "mae"]
    if persistence_mae == 0:
        comparison["mae_vs_persistence_pct"] = np.nan
    else:
        comparison["mae_vs_persistence_pct"] = (
            (persistence_mae - comparison["mae"]) / persistence_mae * 100.0
        )

    if "AEMO P5MIN" in comparison.index:
        aemo_mae = comparison.loc["AEMO P5MIN", "mae"]
        if aemo_mae == 0:
            comparison["mae_vs_aemo_pct"] = np.nan
        else:
            comparison["mae_vs_aemo_pct"] = (
                (aemo_mae - comparison["mae"]) / aemo_mae * 100.0
            )

    return comparison.sort_values("mae")


def compare_models(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    """Compare models and baselines on exactly the same finite test rows.

    Persistence is the current RRP at the forecast origin. When aligned AEMO
    P5MIN forecasts are available, their finite-row mask is also included in
    the shared comparison mask so every value in the table is directly
    comparable.
    """
    actual, shared_mask, forecasts = _comparison_inputs(results, test_data)
    return _comparison_table(actual, forecasts, shared_mask)


def compare_models_by_price_regime(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    """Compare forecasts within regimes defined by the actual future RRP."""
    actual, shared_mask, forecasts = _comparison_inputs(results, test_data)
    comparisons: dict[str, pd.DataFrame] = {}

    for name, lower, upper in PRICE_REGIMES:
        regime_mask = shared_mask.copy()
        if lower is not None:
            regime_mask &= actual >= lower
        if upper is not None:
            regime_mask &= actual < upper
        if regime_mask.any():
            comparisons[name] = _comparison_table(
                actual,
                forecasts,
                regime_mask,
            )

    if not comparisons:
        raise RuntimeError("No shared test rows fall within a price regime")
    return pd.concat(comparisons, names=["regime", "model"])


def compare_models_by_horizon(results: list[ModelResult], test_data: PreparedModelData) -> pd.DataFrame:
    """Reuse scalar comparisons, including available AEMO baselines, per lead."""
    tables = {}
    for position, horizon in enumerate(test_data.forecast_horizons):
        scalar_data = replace(
            test_data, horizon=horizon, forecast_horizons=(),
            test_df=test_data.test_df.assign(target_rrp=test_data.test_df[f"target_rrp_h{horizon:02d}"]),
        )
        scalar_results = [
            ModelResult(r.model_name, r.predictions[:, position], r.valid_mask[:, position], {})
            for r in results
        ]
        tables[horizon] = compare_models(scalar_results, scalar_data)
    return pd.concat(tables, names=["horizon_minutes", "model"])


def _display_table(
    comparison: pd.DataFrame,
    *,
    include_sse: bool,
    include_rows: bool,
) -> str:
    display = comparison.rename(
        columns={
            "mae": "MAE",
            "rmse": "RMSE",
            "sse": "SSE",
            "median_ae": "Median AE",
            "n_rows": "Rows",
            "mae_vs_persistence_pct": "vs Persistence",
            "mae_vs_aemo_pct": "vs AEMO",
            "fit_seconds": "Fit (s)",
        }
    )
    if not include_sse:
        display = display.drop(columns="SSE")
    if not include_rows:
        display = display.drop(columns="Rows")

    formatters = {
        "MAE": lambda value: f"{value:.3f}",
        "RMSE": lambda value: f"{value:.3f}",
        "Median AE": lambda value: f"{value:.3f}",
        "vs Persistence": lambda value: f"{value:+.2f}%",
    }
    if include_sse:
        formatters["SSE"] = lambda value: f"{value:.3e}"
    if include_rows:
        formatters["Rows"] = lambda value: f"{int(value):,}"
    if "vs AEMO" in display.columns:
        formatters["vs AEMO"] = lambda value: f"{value:+.2f}%"
    if "Fit (s)" in display.columns:
        formatters["Fit (s)"] = lambda value: f"{value:.1f}"

    return display.to_string(formatters=formatters, na_rep="n/a")


def print_price_regime_summary(
    results: list[ModelResult],
    test_data: PreparedModelData,
) -> pd.DataFrame:
    """Print model and baseline performance for each actual-price regime."""
    comparisons = compare_models_by_price_regime(results, test_data)

    print("\nRESULTS BY ACTUAL PRICE REGIME")
    print("------------------------------")
    for regime in comparisons.index.get_level_values("regime").unique():
        comparison = comparisons.xs(regime, level="regime")
        row_count = int(comparison["n_rows"].iloc[0])
        print(f"\n{regime} ({row_count:,} rows)")
        print(
            _display_table(
                comparison,
                include_sse=False,
                include_rows=False,
            )
        )
    return comparisons


def print_evaluation_summary(
    results: list[ModelResult],
    test_data: PreparedModelData,
    verbose: bool = True,
    fit_times: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Print a leaderboard; main opts out of legacy detailed tables by default."""
    comparison = compare_models(results, test_data)
    if fit_times:
        comparison["fit_seconds"] = comparison.index.map(fit_times)

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print("All forecasts are scored on the same finite test rows.")
    values = int(comparison["n_rows"].iloc[0])
    print(f"{len(test_data.test_df):,} test origins; {values:,} shared scored forecast values.")
    if getattr(test_data, "forecast_horizons", ()):
        print("Overall errors pool all forecast origins and horizons.")
    display = comparison if verbose else comparison.drop(columns="median_ae")
    print(_display_table(display, include_sse=verbose, include_rows=False))
    print("\nPositive percentages mean lower MAE than the named baseline.")
    if fit_times:
        print("Fit time includes validation search and final refit; excludes GradientSHAP.")
    if verbose:
        print_price_regime_summary(results, test_data)
        if getattr(test_data, "forecast_horizons", ()):
            print("\nTEST RESULTS BY HORIZON ($/MWh)")
            print(_display_table(compare_models_by_horizon(results, test_data), include_sse=False, include_rows=True))
    return comparison
