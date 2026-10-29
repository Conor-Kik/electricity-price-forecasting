"""ARIMA time-series baseline built from a region's own price history.

The model follows the classical two-stage recipe for intraday prices:

1. Optionally compress price spikes with ``asinh`` (a variance-stabilising
   transform that, unlike ``log``, accepts negative prices). ``asinh`` uses
   the fixed scale shared with the neural target transform; ``asinh_mad``
   centres on the median and scales by the MAD, following Uniejewski, Weron
   and Ziel (2018).
2. Remove the deterministic daily/weekly shape with Fourier terms fitted by
   least squares.
3. Fit ARIMA(p, d, q) to what remains, then run the Kalman filter with those
   fixed parameters so every forecast origin uses all prices observed so far.

The forecast for ``t + horizon`` is made at every origin ``t`` from the
filtered state, so it is a rolling forecast, not one forecast of the whole test
period issued at the training cutoff. Parameters are estimated once per
``train`` call, like the other models; only the state is updated afterwards.

``series="aemo_error"`` applies the same machinery to AEMO's realised P5MIN
forecast errors instead of prices, and corrects AEMO's latest forecast with the
predicted error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pipeline.contracts import (
    forecast_array,
    temporal_metadata,
    validate_context,
    validate_timeline,
)
from pipeline.model import Model

TARGET_TYPES = ("change", "price", "aemo_error")
SERIES_TYPES = ("price", "aemo_error")
TRANSFORMS = ("asinh", "asinh_mad", "none")


class ARIMAModel(Model):
    model_name = "ARIMA"
    # Uses the full chronological timeline so the state sees every observation.
    requires_sequence = True

    def __init__(
        self,
        p: int = 2,
        d: int = 0,
        q: int = 1,
        daily_harmonics: int = 4,
        weekly_harmonics: int = 0,
        fit_window_days: float | None = 90,
        transform: str = "asinh",
        asinh_scale: float = 100.0,
        series: str = "price",
        horizon_minutes: int = 30,
        target_type: str = "change",
        interval_minutes: int = 5,
        context_length: int = 2016,
        max_iter: int = 100,
        forecast_horizons: tuple[int, ...] = (),
    ) -> None:
        for name, value in (("p", p), ("d", d), ("q", q)):
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if d > 2:
            raise ValueError("d above 2 is not supported")
        if (
            int(daily_harmonics) != daily_harmonics
            or int(weekly_harmonics) != weekly_harmonics
            or daily_harmonics < 0
            or weekly_harmonics < 0
        ):
            raise ValueError("Fourier harmonics must be non-negative integers")
        if fit_window_days is not None and fit_window_days <= 0:
            raise ValueError("fit_window_days must be positive or None")
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        if asinh_scale <= 0:
            raise ValueError("asinh_scale must be positive")
        if series not in SERIES_TYPES:
            raise ValueError(f"series must be one of {SERIES_TYPES}")
        if target_type not in TARGET_TYPES:
            raise ValueError(f"target_type must be one of {TARGET_TYPES}")
        if interval_minutes <= 0 or horizon_minutes <= 0:
            raise ValueError("horizon_minutes and interval_minutes must be positive")
        if horizon_minutes % interval_minutes:
            raise ValueError("horizon_minutes must be a multiple of interval_minutes")
        if context_length < 1 or max_iter < 1:
            raise ValueError("context_length and max_iter must be positive")

        self.p, self.d, self.q = int(p), int(d), int(q)
        self.daily_harmonics = int(daily_harmonics)
        self.weekly_harmonics = int(weekly_harmonics)
        self.fit_window_days = fit_window_days
        self.transform = transform
        self.asinh_scale = float(asinh_scale)
        self.series = series
        self.horizon_minutes = int(horizon_minutes)
        self.target_type = target_type
        self.interval_minutes = int(interval_minutes)
        self.steps = self.horizon_minutes // self.interval_minutes
        self.forecast_horizons = tuple(forecast_horizons)
        if self.forecast_horizons:
            if (tuple(sorted(set(self.forecast_horizons))) != self.forecast_horizons
                    or any(h <= 0 or h % self.interval_minutes for h in self.forecast_horizons)
                    or max(self.forecast_horizons) != self.horizon_minutes):
                raise ValueError("forecast_horizons must be increasing intervals ending at horizon_minutes")
            if series == "aemo_error":
                raise ValueError("Multi-horizon ARIMA requires series='price'; AEMO residual series are horizon-specific")
        # Kalman burn-in before the first prediction; one week by default.
        self.context_length = max(int(context_length), self.steps + 1)
        self.max_iter = int(max_iter)

        self.params = None
        self.transform_center, self.transform_scale = 0.0, self.asinh_scale
        self.seasonal_coefficients = None
        self.context_frame = None
        self.converged = None
        self.aic = None
        self.bic = None
        self.fit_rows = 0
        self.fallback_rows = 0

    @property
    def aemo_column(self) -> str:
        return f"p5_h{self.horizon_minutes:02d}_rrp"

    def _arima(self, values: np.ndarray) -> SARIMAX:
        return SARIMAX(
            values,
            order=(self.p, self.d, self.q),
            trend="n",
            concentrate_scale=True,
        )

    def _fit_transform(self, values: np.ndarray) -> None:
        self.transform_center, self.transform_scale = 0.0, self.asinh_scale
        if self.transform == "asinh_mad":
            self.transform_center = float(np.median(values))
            # 1.4826 makes the MAD consistent with the standard deviation.
            mad = 1.4826 * float(np.median(np.abs(values - self.transform_center)))
            if mad > 0:
                self.transform_scale = mad

    def _forward(self, values):
        if self.transform == "none":
            return values
        return np.arcsinh((values - self.transform_center) / self.transform_scale)

    def _inverse(self, values):
        if self.transform == "none":
            return values
        return self.transform_center + self.transform_scale * np.sinh(values)

    def _timestamps(self, frame: pd.DataFrame) -> pd.DatetimeIndex | None:
        metadata = temporal_metadata(frame)
        if metadata is not None:
            validate_timeline(metadata, self.interval_minutes)
            return pd.DatetimeIndex(metadata["settlement_date"])
        if self.daily_harmonics or self.weekly_harmonics:
            raise ValueError("Fourier seasonal terms need timestamped inputs")
        return None

    def _seasonal_design(self, times: pd.DatetimeIndex | None, count: int) -> np.ndarray:
        """Intercept plus sine/cosine pairs for the daily and weekly cycles."""
        columns = [np.ones(count)]
        if times is not None:
            day_fraction = (times.hour * 60 + times.minute).to_numpy() / 1440.0
            week_fraction = (times.dayofweek.to_numpy() + day_fraction) / 7.0
            for fraction, harmonics in (
                (day_fraction, self.daily_harmonics),
                (week_fraction, self.weekly_harmonics),
            ):
                for k in range(1, harmonics + 1):
                    angle = 2 * np.pi * k * fraction
                    columns.extend((np.sin(angle), np.cos(angle)))
        return np.column_stack(columns)

    def _numeric(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        if column not in frame.columns:
            raise ValueError(f"ARIMA requires the {column!r} feature")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, np.nan)

    def _shift(self, values: np.ndarray, steps: int) -> np.ndarray:
        """Positional shift; validated timelines make positions equal intervals."""
        shifted = np.full(len(values), np.nan)
        if steps > 0:
            shifted[steps:] = values[:-steps]
        elif steps < 0:
            shifted[:steps] = values[-steps:]
        else:
            shifted[:] = values
        return shifted

    def _observed_series(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the modelled series as known at each row's own timestamp."""
        price = self._numeric(frame, "rrp")
        if self.series == "price":
            return price
        # AEMO's forecast of this interval was issued `steps` rows earlier.
        return price - self._shift(self._numeric(frame, self.aemo_column), self.steps)

    def _check_targets(self, frame: pd.DataFrame, targets: pd.Series | pd.DataFrame) -> None:
        """Fail loudly when horizon/target settings disagree with the experiment."""
        price = self._numeric(frame, "rrp")
        horizons = self.forecast_horizons or (self.horizon_minutes,)
        if self.forecast_horizons and (not isinstance(targets, pd.DataFrame) or targets.shape[1] != len(horizons)):
            raise ValueError("Multi-horizon ARIMA needs one target column per declared horizon")
        for position, horizon in enumerate(horizons):
            future = self._shift(price, -(horizon // self.interval_minutes))
            if self.target_type == "change":
                implied = future - price
            elif self.target_type == "price":
                implied = future
            else:
                implied = future - self._numeric(frame, f"p5_h{horizon:02d}_rrp")
            labels = targets.iloc[:, position] if isinstance(targets, pd.DataFrame) else targets
            actual = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=float)
            comparable = np.isfinite(actual) & np.isfinite(implied)
            if comparable.any() and not np.allclose(implied[comparable], actual[comparable], rtol=1e-9, atol=1e-6):
                raise ValueError(f"Training targets do not match horizon_minutes={horizon} and target_type={self.target_type!r}")

    def train(self, x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame) -> None:
        if x_train.empty or len(x_train) != len(y_train):
            raise ValueError(
                "Training inputs and targets must be nonempty and have equal lengths"
            )
        times = self._timestamps(x_train)
        self._check_targets(x_train, y_train)

        values = self._observed_series(x_train)
        if self.fit_window_days is not None and times is not None:
            # Estimate on recent data only; prices may not be stationary across years.
            window = np.asarray(times > times[-1] - pd.Timedelta(days=self.fit_window_days))
            values, times = values[window], times[window]
        design = self._seasonal_design(times, len(values))
        finite = np.isfinite(values)
        if finite.sum() <= max(self.p, self.q) + self.d + design.shape[1]:
            raise ValueError("Not enough finite observations to fit ARIMA")

        self._fit_transform(values[finite])
        values = self._forward(values)
        self.seasonal_coefficients = np.linalg.lstsq(
            design[finite], values[finite], rcond=None
        )[0]
        residual = values - design @ self.seasonal_coefficients
        arima = self._arima(residual)
        if arima.k_params == 0:
            # ARIMA(0, d, 0) has nothing to estimate; d=1 is the random walk.
            self.params = np.empty(0)
            self.converged = True
            result = arima.filter(self.params)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = arima.fit(disp=False, maxiter=self.max_iter)
            self.params = np.asarray(result.params, dtype=float)
            self.converged = bool(result.mle_retvals.get("converged", True))
        self.aic, self.bic = float(result.aic), float(result.bic)
        self.fit_rows = int(finite.sum())
        self.context_frame = x_train.tail(self.context_length).copy()

    def _state_forecast(self, residual: np.ndarray) -> np.ndarray:
        """Expected residual `steps` ahead from the filtered state at every row."""
        arima = self._arima(residual)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = arima.filter(self.params).filtered_state
        system, states = arima.ssm, arima.k_states
        # The system is time-invariant; drop any trailing time dimension.
        transition = np.reshape(system["transition"], (states, states))
        design = np.reshape(system["design"], (1, states))
        state_intercept = np.reshape(system["state_intercept"], states)
        obs_intercept = np.reshape(system["obs_intercept"], 1)
        # Filter once, then advance each origin's state without observing any
        # future prices. All requested leads come from this same fitted model.
        leads = {h // self.interval_minutes for h in (self.forecast_horizons or (self.horizon_minutes,))}
        ahead, forecasts = filtered, []
        for step in range(1, self.steps + 1):
            ahead = transition @ ahead + state_intercept[:, None]
            if step in leads:
                forecasts.append((design @ ahead + obs_intercept[:, None])[0])
        return np.column_stack(forecasts) if self.forecast_horizons else forecasts[0]

    def predict(self, x_test: pd.DataFrame, *, history=None) -> np.ndarray:
        """Forecast `horizon_minutes` ahead from every row, in target units.

        ``history`` holds observed rows immediately before ``x_test``; without
        it, the stored training tail is used. Neither is modified.
        """
        if self.params is None:
            raise RuntimeError("train() must be called before predict()")
        if x_test.empty:
            return np.empty((0, len(self.forecast_horizons))) if self.forecast_horizons else np.empty(0, dtype=float)
        context = (self.context_frame if history is None else history).tail(
            self.context_length
        )
        validate_context(context, x_test, self.interval_minutes)
        combined = pd.concat([context.reindex(columns=x_test.columns), x_test])
        times = self._timestamps(combined)

        observed = self._forward(self._observed_series(combined))
        count = len(combined)
        residual = observed - self._seasonal_design(times, count) @ self.seasonal_coefficients
        seasonal = [
            self._seasonal_design(None if times is None else times + pd.Timedelta(minutes=h), count)
            @ self.seasonal_coefficients
            for h in (self.forecast_horizons or (self.horizon_minutes,))
        ]
        seasonal_ahead = np.column_stack(seasonal) if self.forecast_horizons else seasonal[0]
        forecast = self._inverse(self._state_forecast(residual) + seasonal_ahead)

        current = slice(count - len(x_test), count)
        price = self._numeric(combined, "rrp")[current]
        forecast = forecast[current]
        if self.forecast_horizons:
            price = price[:, None]
        if self.series == "aemo_error":
            aemo = self._numeric(combined, self.aemo_column)[current]
            missing = ~np.isfinite(aemo)
            # Without AEMO's latest forecast, fall back to persistence.
            forecast = np.where(missing, price, aemo + forecast)
            self.fallback_rows = int(missing.sum())

        if self.target_type == "change":
            forecast = forecast - price
        elif self.target_type == "aemo_error":
            aemo = (np.column_stack([self._numeric(x_test, f"p5_h{h:02d}_rrp") for h in self.forecast_horizons])
                    if self.forecast_horizons else self._numeric(x_test, self.aemo_column))
            forecast = forecast - aemo
        return forecast_array(forecast, len(x_test), len(self.forecast_horizons) or 1)
