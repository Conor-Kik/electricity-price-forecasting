from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for essential NEM price-prediction features.

    All lag/window values are in 5-minute intervals by default. The defaults are
    intentionally compact: they capture immediate dynamics, intra-hour behaviour,
    and the same time on the previous day without exploding feature count.
    """

    interval_minutes: int = 5

    # Forecast target. Additional target horizons can be added later without
    # changing the feature engineering code.
    target_horizons_minutes: tuple[int, ...] = (5,)

    # Historical signal.
    price_lags: tuple[int, ...] = (1, 2, 3, 6, 12, 36, 288)
    demand_lags: tuple[int, ...] = (1, 3, 6, 12, 288)
    delta_intervals: tuple[int, ...] = (1, 3, 6, 12)
    rolling_windows: tuple[int, ...] = (6, 12, 36, 288)

    # Full AEMO P5MIN trajectory: all twelve leads, available at the forecast origin.
    p5min_horizons_minutes: tuple[int, ...] = tuple(range(5, 61, 5))

    include_calendar: bool = True
    include_interconnector_utilisation: bool = True

    # Forecast reliability: how accurate AEMO's own P5MIN price forecast has been
    # recently. These are the forecast horizons that are scored, the windows the
    # errors are averaged over, and the price that counts as a spike.
    reliability_horizons_minutes: tuple[int, ...] = (5, 30)
    reliability_windows: tuple[int, ...] = (12, 72, 288)
    reliability_spike_level: float = 300.0

    # Optional clipping is deliberately disabled by default. Extreme prices are
    # important observations in the NEM and should not be silently removed.
    clip_infinite_ratios: bool = True

    @property
    def max_history_intervals(self) -> int:
        # A reliability window also needs the forecasts issued one horizon earlier.
        reliability = max(self.reliability_windows, default=0) + (
            max(self.reliability_horizons_minutes, default=0) // self.interval_minutes
        )
        values = [
            1,
            *self.price_lags,
            *self.demand_lags,
            *self.rolling_windows,
            reliability,
        ]
        return max(values)

    @property
    def max_history_minutes(self) -> int:
        return self.max_history_intervals * self.interval_minutes
