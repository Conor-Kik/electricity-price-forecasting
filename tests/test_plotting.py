"""Regression checks for interactive evaluation plots using an offscreen canvas."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, NavigationToolbar2
import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation import ModelResult
from pipeline.plotting import plot_price_regimes


@pytest.mark.parametrize("model_error", [0.0, 10.0, 20.0])
def test_price_regime_hover_at_colour_limits(model_error):
    actual = np.array([-10.0, 100.0, 500.0, 1500.0])
    prepared = SimpleNamespace(
        horizon=30,
        test_df=pd.DataFrame({"target_rrp": actual, "rrp": actual - 10.0}),
    )
    result = ModelResult(
        model_name="ARIMA",
        predictions=actual + model_error,
        valid_mask=np.ones(len(actual), dtype=bool),
        metrics={},
    )
    figure = plot_price_regimes([result], prepared)
    try:
        figure.canvas.draw()
        axis = figure.axes[0]
        image = axis.images[0]
        improvement = 100.0 * (1.0 - model_error / 10.0)
        np.testing.assert_allclose(image.get_array(), improvement)
        # Preserve a symmetric scale with zero at the neutral colour.
        limit = max(abs(improvement), 1.0)
        np.testing.assert_allclose(image.norm([-limit, 0.0, limit]), [0.0, 0.5, 1.0])

        messages = []
        toolbar = NavigationToolbar2(figure.canvas)
        toolbar.set_message = messages.append
        for column in range(len(actual)):
            x, y = axis.transData.transform((column, 0))
            event = MouseEvent("motion_notify_event", figure.canvas, x, y)
            assert event.inaxes is axis
            # Exercise the toolbar path from the reported traceback. With
            # TwoSlopeNorm, a +100% cell raises OverflowError here.
            toolbar.mouse_move(event)
            assert messages[-1]
            assert "inf" not in messages[-1].lower()
    finally:
        plt.close(figure)
