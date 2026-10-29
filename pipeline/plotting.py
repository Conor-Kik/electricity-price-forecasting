"""Plots reuse the evaluation tables so their metrics share the console's masks."""

from pathlib import Path

import numpy as np

from pipeline.contracts import price_values
from pipeline.evaluation import (
    compare_models_by_horizon,
    compare_models_by_price_regime,
)

ATTRIBUTION_LABEL = "Share of total absolute attribution (%)"


def _styles(results):
    styles = {r.model_name: {"color": f"C{i % 10}", "linestyle": "-"}
              for i, r in enumerate(results)}
    styles["Persistence"] = {"color": "0.45", "linestyle": "--"}
    styles["AEMO P5MIN"] = {"color": "0.15", "linestyle": ":"}
    return styles


def plot_horizon_errors(results, prepared):
    """Show MAE and RMSE together, leaving unavailable baseline leads as gaps."""
    import matplotlib.pyplot as plt

    horizons = prepared.forecast_horizons
    table = compare_models_by_horizon(results, prepared)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for name, style in _styles(results).items():
        if name not in table.index.get_level_values("model"):
            continue
        values = table.xs(name, level="model").reindex(horizons)
        for axis, metric in zip(axes, ("mae", "rmse")):
            axis.plot(horizons, values[metric], marker="o", markersize=4, label=name, **style)
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.set(title=metric, xlabel="Minutes ahead", ylabel=f"{metric} ($/MWh)", xticks=horizons)
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=9)
    figure.suptitle("Held-out forecast errors · lower is better")
    return figure


def _top_features(importance, count):
    ranked = importance.mean(axis=1).sort_values(ascending=False, kind="stable")
    return ranked.index[:count] if count else ranked.index


def _horizon_slider(figure, horizons, draw, *, interactive, width):
    """Share slider setup and keep its callback alive for both feature views."""
    from matplotlib.widgets import Slider

    if interactive and len(horizons) > 1:
        slider_axis = figure.add_axes([0.29, 0.07, width, 0.035])
        slider = Slider(slider_axis, "Forecast (min)", horizons[0], horizons[-1],
                        valinit=horizons[-1], valstep=np.asarray(horizons), valfmt="+%0.0f min")
        slider.on_changed(draw)
        figure._horizon_slider = slider
    draw(horizons[-1])


def _importance_heatmap(axis, frame, *, title, xlabel, ylabel, maximum=None,
                        column_labels=None, row_labels=None, thin_rows=False):
    """Draw every cell, thinning only tick labels for long sequence windows."""
    maximum = float(frame.max().max()) if maximum is None else maximum
    image = axis.imshow(np.ma.masked_invalid(frame.to_numpy()), aspect="auto",
                        interpolation="nearest", cmap="viridis", vmin=0, vmax=maximum or 1)
    columns = np.unique(np.linspace(0, len(frame.columns) - 1, min(16, len(frame.columns)), dtype=int))
    rows = (np.unique(np.linspace(0, len(frame) - 1, min(20, len(frame)), dtype=int))
            if thin_rows else np.arange(len(frame)))
    column_labels = list(frame.columns) if column_labels is None else column_labels
    row_labels = list(frame.index) if row_labels is None else row_labels
    axis.set(xticks=columns, xticklabels=[column_labels[i] for i in columns],
             yticks=rows, yticklabels=[row_labels[i] for i in rows],
             title=title, xlabel=xlabel, ylabel=ylabel)
    axis.tick_params(labelsize=9)
    if frame.size <= 200:
        for row, column in np.ndindex(frame.shape):
            value = frame.iloc[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                          color="white" if value < maximum * 0.55 else "black")

    # The native plot toolbar shows exact values without crowding dense grids.
    def coordinate(x, y):
        column, row = int(round(x)), int(round(y))
        if 0 <= row < len(frame) and 0 <= column < len(frame.columns):
            return f"{row_labels[row]} | {column_labels[column]}: {frame.iloc[row, column]:.2f}% of total attribution"
        return ""

    axis.format_coord = coordinate
    return image


def plot_shap_heatmap(importance, model_name, *, maximum=None, top_features=20,
                      sequence_importance=None, sequence_maximum=None):
    """Feature and sequence-step percentage overviews across forecast leads."""
    import matplotlib.pyplot as plt

    top = _top_features(importance, top_features)
    has_sequence = sequence_importance is not None and not sequence_importance.empty
    figure, axes = plt.subplots(1, 2 if has_sequence else 1, squeeze=False,
                               figsize=(17 if has_sequence else 11, max(5, 0.27 * len(top) + 2.5)),
                               layout="constrained")
    figure.canvas.manager.set_window_title(f"{model_name}: GradientSHAP")
    axis = axes[0, 0]
    image = _importance_heatmap(axis, importance.loc[top], title="Feature importance (summed over sequence steps)",
                                xlabel="Forecast horizon (minutes ahead)", ylabel="Feature", maximum=maximum,
                                column_labels=[f"+{h}" for h in importance.columns])
    figure.colorbar(image, ax=axis, label=ATTRIBUTION_LABEL)
    if has_sequence:
        axis = axes[0, 1]
        image = _importance_heatmap(axis, sequence_importance, title="Sequence importance (summed over features)",
                                    xlabel="Forecast horizon (minutes ahead)", ylabel="Minutes before origin (0 = latest)",
                                    maximum=sequence_maximum, thin_rows=True,
                                    column_labels=[f"+{h}" for h in sequence_importance.columns])
        figure.colorbar(image, ax=axis, label=ATTRIBUTION_LABEL)
    figure.suptitle(f"{model_name}: GradientSHAP shares per forecast horizon")
    return figure


def plot_feature_bars(importance, model_name, *, top_features=20, interactive=True, maximum=None):
    """Rank non-sequence percentage inputs separately for each forecast target."""
    import matplotlib.pyplot as plt

    horizons = sorted(importance.columns)
    count = min(top_features, len(importance)) if top_features else len(importance)
    maximum = float(importance.max().max()) if maximum is None else maximum
    figure = plt.figure(figsize=(12, max(5, 0.27 * count + 2.8)))
    figure.canvas.manager.set_window_title(f"{model_name}: top features")
    axis = figure.add_axes([0.29, 0.23, 0.65, 0.63])

    def draw(horizon):
        values = importance[int(horizon)].sort_values(ascending=False, kind="stable").iloc[:count]
        axis.clear()
        axis.barh(np.arange(len(values)), values.to_numpy())
        axis.set(yticks=np.arange(len(values)), yticklabels=values.index,
                 xlim=(0, (maximum or 1) * 1.1), title=f"Forecast +{int(horizon)} min",
                 xlabel=ATTRIBUTION_LABEL)
        axis.invert_yaxis()
        axis.tick_params(labelsize=9)
        axis.grid(axis="x", alpha=0.2)
        figure.canvas.draw_idle()

    figure.suptitle(f"{model_name}: top features for the selected forecast target")
    _horizon_slider(figure, horizons, draw, interactive=interactive, width=0.65)
    return figure


def plot_feature_sequence(summary, model_name, *, top_features=20, interactive=True, maximum=None):
    """Slice feature x lookback-step percentage shares by forecast horizon."""
    import matplotlib.pyplot as plt

    count = min(top_features, len(summary.features)) if top_features else len(summary.features)
    horizons = sorted(summary.feature_lags)
    if maximum is None:
        maximum = max(float(frame.max().max()) for frame in summary.feature_lags.values())
    figure = plt.figure(figsize=(12, max(5, 0.27 * count + 2.8)))
    figure.canvas.manager.set_window_title(f"{model_name}: features across sequence")
    axis = figure.add_axes([0.29, 0.23, 0.57, 0.63])
    colour_axis = figure.add_axes([0.89, 0.23, 0.025, 0.63])

    def draw(horizon):
        horizon = int(horizon)
        # Rank this lead's features by total importance across the sequence.
        # Keep the colour scale fixed so magnitudes remain comparable across leads.
        frame = summary.feature_lags[horizon]
        ranked = frame.sum(axis=1).sort_values(ascending=False, kind="stable").index
        selected = ranked[:top_features] if top_features else ranked
        frame = frame.loc[selected].sort_index(axis=1, ascending=False)
        axis.clear()
        colour_axis.clear()
        image = _importance_heatmap(axis, frame, title=f"Forecast +{horizon} min",
                                    xlabel="Sequence time relative to origin (minutes)", ylabel="Feature", maximum=maximum,
                                    column_labels=["0 (latest)" if lag == 0 else f"−{lag}" for lag in frame.columns])
        figure.colorbar(image, cax=colour_axis, label=ATTRIBUTION_LABEL)
        figure.canvas.draw_idle()

    figure.suptitle(f"{model_name}: feature importance across the observed sequence")
    _horizon_slider(figure, horizons, draw, interactive=interactive, width=0.57)
    return figure


def plot_sequence_surface(sequence_importance, model_name, *, maximum=None):
    """Optional rotatable surface of forecast lead x past step, summed over features."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 7), layout="constrained")
    figure.canvas.manager.set_window_title(f"{model_name}: 3D sequence importance")
    axis = figure.add_subplot(projection="3d")
    x, y = np.meshgrid(sequence_importance.columns, sequence_importance.index)
    values = sequence_importance.to_numpy()
    limits = {"vmin": 0, "vmax": maximum or max(float(np.nanmax(values)), 1e-8)}
    if min(values.shape) > 1:
        image = axis.plot_surface(x, y, values, cmap="viridis", rcount=len(values),
                                  ccount=values.shape[1], **limits)
    else:
        image = axis.scatter(x.ravel(), y.ravel(), values.ravel(), c=values.ravel(), cmap="viridis", **limits)
    axis.set(xlabel="Forecast horizon (min ahead)", ylabel="Minutes before origin",
             zlabel="Share of total attribution (%)", title=f"{model_name}: sequence attribution shares per forecast horizon")
    figure.colorbar(image, ax=axis, shrink=0.6, pad=0.12, label=ATTRIBUTION_LABEL)
    return figure


def plot_price_regimes(results, prepared):
    """Show where each model improves or deteriorates relative to persistence."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    scores = compare_models_by_price_regime(results, prepared)
    values = scores["mae_vs_persistence_pct"].unstack("regime")
    values = values.reindex(columns=scores.index.get_level_values("regime").unique()).drop(index="Persistence")
    finite = values.to_numpy()[np.isfinite(values.to_numpy())]
    limit = max(float(np.abs(finite).max()), 1.0) if finite.size else 1.0
    figure, axis = plt.subplots(figsize=(11, max(3.5, 0.5 * len(values))), layout="constrained")
    # Symmetric limits put zero at the midpoint with ordinary linear scaling.
    # TwoSlopeNorm's infinite inverse outside [0, 1] breaks Matplotlib's hover
    # precision calculation for cells at the upper colour limit.
    image = axis.imshow(np.ma.masked_invalid(values.to_numpy()), cmap="RdYlGn",
                        norm=Normalize(vmin=-limit, vmax=limit), aspect="auto")
    axis.set(xticks=np.arange(len(values.columns)), xticklabels=[c.split(" (")[0] for c in values.columns],
             yticks=np.arange(len(values)), yticklabels=values.index,
             title="MAE improvement over persistence by actual price regime")
    for row, column in np.ndindex(values.shape):
        value = values.iloc[row, column]
        if np.isfinite(value):
            axis.text(column, row, f"{value:+.1f}%", ha="center", va="center",
                      color="white" if abs(value) > limit * 0.7 else "black")
    figure.colorbar(image, ax=axis, label="Improvement (%) · positive is better")
    return figure


def plot_forecast_examples(results, prepared, *, seed=42, count=6):
    """Sample shared test origins without using model errors or future prices to rank them."""
    import matplotlib.pyplot as plt

    horizons = prepared.forecast_horizons
    actual = price_values(prepared.test_df, prepared.forecast_horizons).reshape(len(prepared.test_df), -1)
    eligible = np.isfinite(actual).all(axis=1) & np.isfinite(prepared.test_df["rrp"].to_numpy())
    for result in results:
        eligible &= result.valid_mask.reshape(len(actual), -1).all(axis=1)
        eligible &= np.isfinite(result.predictions.reshape(len(actual), -1)).all(axis=1)
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return None
    selected = np.sort(np.random.default_rng(seed).choice(rows, size=min(count, len(rows)), replace=False))
    columns = min(3, len(selected))
    panels = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(panels, columns, figsize=(5 * columns, 3.8 * panels), squeeze=False, layout="constrained")
    styles = _styles(results)
    for axis, row in zip(axes.flat, selected):
        current = prepared.test_df.iloc[row]["rrp"]
        leads = (0, *horizons)
        axis.plot(leads, np.r_[current, actual[row]], color="black", linewidth=2, label="Actual")
        for result in results:
            values = result.predictions.reshape(len(actual), -1)[row]
            axis.plot(leads, np.r_[current, values], label=result.model_name, **styles[result.model_name])
        axis.axhline(current, label="Persistence", **styles["Persistence"])
        columns = [f"p5_h{h:02d}_rrp" for h in horizons]
        aemo = prepared.test_df.reindex(columns=columns).iloc[row].to_numpy(dtype=float)
        if np.isfinite(aemo).any():
            axis.plot(leads, np.r_[current, aemo], label="AEMO P5MIN", **styles["AEMO P5MIN"])
        axis.set(title=str(prepared.test_df.iloc[row]["settlement_date"]), xlabel="Minutes ahead", ylabel="Price ($/MWh)")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(selected):]:
        axis.set_visible(False)
    axes.flat[len(selected) - 1].legend(fontsize=8)
    figure.suptitle(f"Price reconstruction: {len(selected)} randomly sampled test origins (seed {seed})")
    return figure


def present_results(results, prepared, importances, *, show=False, directory=None, details=False,
                    top_features=20, sequence_3d=False, seed=42):
    """Display only on request; export only to an explicitly enabled run folder."""
    import matplotlib.pyplot as plt

    def present(figure, filename):
        if figure is None:
            return
        if directory is not None:
            figure.savefig(Path(directory) / filename, dpi=150, bbox_inches="tight")
        if not show:
            plt.close(figure)

    present(plot_horizon_errors(results, prepared), "errors_by_horizon.png")
    present(plot_forecast_examples(results, prepared, seed=seed), "forecast_examples.png")
    # Normalise before ranking or selecting features; hidden features still count.
    importances = {name: summary.as_percent() for name, summary in importances.items()}
    maximum = max((float(summary.features.max().max()) for summary in importances.values()), default=1.0)
    sequence_maximum = max((float(summary.lags.max().max()) for summary in importances.values() if summary.feature_lags), default=1.0)
    cell_maximum = max((float(frame.max().max()) for summary in importances.values()
                        for frame in summary.feature_lags.values()), default=1.0)
    for name, summary in importances.items():
        slug = name.lower().replace(" ", "-")
        if summary.feature_lags:
            present(plot_shap_heatmap(summary.features, name, maximum=maximum, top_features=top_features,
                                     sequence_importance=summary.lags, sequence_maximum=sequence_maximum),
                    f"{slug}/gradient_shap_heatmap.png")
        else:
            present(plot_feature_bars(summary.features, name, maximum=maximum,
                                      top_features=top_features, interactive=show),
                    f"{slug}/gradient_shap_top_features.png")
        if summary.feature_lags:
            horizon = max(summary.feature_lags)
            present(plot_feature_sequence(summary, name, top_features=top_features, interactive=show, maximum=cell_maximum),
                    f"{slug}/gradient_shap_feature_sequence_h{horizon:02d}.png")
            if sequence_3d:
                present(plot_sequence_surface(summary.lags, name, maximum=sequence_maximum),
                        f"{slug}/gradient_shap_sequence_3d.png")
    if details:
        present(plot_price_regimes(results, prepared), "errors_by_price_regime.png")
    if show:
        plt.show()
