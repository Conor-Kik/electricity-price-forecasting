from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline.evaluation import (
    ModelResult,
    compare_models,
    compare_models_by_price_regime,
    print_evaluation_summary,
)


def result(name, predictions, valid_mask=None):
    predictions = np.asarray(predictions, dtype=float)
    if valid_mask is None:
        valid_mask = np.isfinite(predictions)
    return ModelResult(
        model_name=name,
        predictions=predictions,
        valid_mask=np.asarray(valid_mask, dtype=bool),
        metrics={},
    )


def test_compare_models_uses_shared_model_and_aemo_rows():
    test_data = SimpleNamespace(
        horizon=30,
        test_df=pd.DataFrame(
            {
                "target_rrp": [100.0, 200.0, 300.0, 400.0],
                "rrp": [90.0, 210.0, 290.0, 380.0],
                "p5_h30_rrp": [95.0, np.nan, 280.0, 390.0],
            }
        ),
    )
    results = [
        result("Model A", [110.0, 190.0, 310.0, 410.0]),
        result("Model B", [100.0, 200.0, np.nan, 420.0]),
    ]

    comparison = compare_models(results, test_data)

    assert set(comparison.index) == {
        "Model A",
        "Model B",
        "Persistence",
        "AEMO P5MIN",
    }
    assert comparison["n_rows"].eq(2).all()
    assert comparison.loc["Model A", "mae"] == 10.0
    assert comparison.loc["Model B", "mae"] == 10.0
    assert comparison.loc["Persistence", "mae"] == 15.0
    assert comparison.loc["AEMO P5MIN", "mae"] == 7.5


def test_compare_models_by_actual_price_regime_uses_expected_boundaries():
    test_data = SimpleNamespace(
        horizon=30,
        test_df=pd.DataFrame(
            {
                "target_rrp": [-1.0, 0.0, 299.0, 300.0, 999.0, 1000.0],
                "rrp": [-2.0, 2.0, 295.0, 290.0, 980.0, 950.0],
                "p5_h30_rrp": [-3.0, 1.0, 300.0, 305.0, 1009.0, 1020.0],
            }
        ),
    )
    results = [result("MLP", [-1.0, 1.0, 297.0, 304.0, 1004.0, 1010.0])]

    comparisons = compare_models_by_price_regime(results, test_data)

    expected_rows = {
        "Negative (< $0/MWh)": 1,
        "Normal ($0-$300/MWh)": 2,
        "High ($300-$1,000/MWh)": 2,
        "Extreme (>= $1,000/MWh)": 1,
    }
    assert set(comparisons.index.get_level_values("regime")) == set(expected_rows)
    for regime, count in expected_rows.items():
        assert comparisons.xs(regime)["n_rows"].eq(count).all()

    extreme = comparisons.xs("Extreme (>= $1,000/MWh)")
    assert extreme.loc["MLP", "mae"] == 10.0
    assert extreme.loc["Persistence", "mae"] == 50.0
    assert extreme.loc["MLP", "mae_vs_persistence_pct"] == 80.0


def test_print_evaluation_summary_does_not_dump_predictions(capsys):
    test_data = SimpleNamespace(
        horizon=30,
        test_df=pd.DataFrame(
            {
                "target_rrp": [100.0, 200.0],
                "rrp": [90.0, 190.0],
            }
        ),
    )

    print_evaluation_summary(
        [result("MLP", [101.123456789, 198.0])],
        test_data,
    )

    output = capsys.readouterr().out
    assert "FINAL TEST RESULTS" in output
    assert "RESULTS BY ACTUAL PRICE REGIME" in output
    assert "Normal ($0-$300/MWh) (2 rows)" in output
    assert "Persistence" in output
    assert "101.123456789" not in output
    assert "predictions=array" not in output
