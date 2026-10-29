"""Small attribution aggregates for plots, keeping the two time axes separate."""

from dataclasses import dataclass, field

import pandas as pd


def importance_percent(values: pd.DataFrame) -> pd.DataFrame:
    """Shares of all absolute attribution in each horizon, before top-N filtering.

    A horizon with no attribution stays zero rather than dividing by zero.
    These are shares of aggregate magnitude, not explained variance or accuracy.
    """
    totals = values.sum(axis=0)
    return values.div(totals.where(totals > 0), axis=1).fillna(0) * 100


@dataclass
class ShapSummary:
    features: pd.DataFrame
    # Forecast horizon -> feature rows x minutes-before-origin columns.
    feature_lags: dict[int, pd.DataFrame] = field(default_factory=dict)

    @property
    def lags(self) -> pd.DataFrame:
        """Sum absolute feature contributions at each observed sequence step."""
        return pd.DataFrame({h: frame.sum(axis=0) for h, frame in self.feature_lags.items()}).sort_index()

    def as_percent(self) -> "ShapSummary":
        """Use the same all-feature/all-step denominator for each horizon."""
        totals = self.features.sum(axis=0)
        return ShapSummary(
            importance_percent(self.features),
            {h: frame * (100 / totals[h] if totals[h] > 0 else 0)
             for h, frame in self.feature_lags.items()},
        )

    def diagnostics(self) -> pd.DataFrame:
        """Describe feature concentration and input-row lookbacks.

        Engineered inputs (for example rrp_lag_288) may already encode older
        information, even when supplied in the latest sequence row.
        """
        relative = self.as_percent()
        features = relative.features
        records = {
            "P5 share (%)": features.loc[features.index.str.startswith("p5_")].sum(),
            "Top 5 share (%)": features.apply(lambda column: column.nlargest(5).sum()),
        }
        if self.feature_lags:
            lags = relative.lags.sort_index()
            records["Latest row (%)"] = lags.loc[0]
            # Five-minute data: 0, 5, ..., 25 are the six most recent observations.
            records["Sequence <30 min (%)"] = lags.loc[lags.index < 30].sum()
            # Smallest lookback including 90% of the total absolute attribution.
            records["90% sequence lookback (min)"] = lags.cumsum().apply(
                lambda column: column.index[column >= 90 - 1e-8][0]
                if (column >= 90 - 1e-8).any() else float("nan")
            )
        return pd.DataFrame(records).rename_axis("horizon_minutes")


def feature_lag_importance(attribution: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Average absolute contributions before aggregation, avoiding cancellation.

    The existing GradientSHAP output names sequence positions lag_0, lag_1, etc.
    These positions are distinct from features such as rrp_lag_288.
    """
    values = attribution.attrs["lag_attributions"].abs().mean()
    index = pd.MultiIndex.from_tuples(
        [(feature, int(lag) * interval_minutes) for _, lag, feature in
         (column.split("_", 2) for column in values.index)],
        names=["feature", "minutes_before_prediction"],
    )
    return pd.Series(values.to_numpy(), index=index).unstack().sort_index(axis=1, ascending=False)
