from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .schema import COLUMN_ALIASES, CORE_ACTUAL_COLUMNS, P5MIN_VALUE_COLUMNS


def canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with common AEMO columns mapped to stable lowercase names."""
    out = df.copy()
    rename = {c: COLUMN_ALIASES.get(c, c.lower()) for c in out.columns}
    out = out.rename(columns=rename)

    for col in ("settlement_date", "run_datetime", "forecast_time", "lastchanged"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    if "region" in out.columns:
        out["region"] = out["region"].astype(str)
    return out


def validate_actuals(df: pd.DataFrame) -> None:
    missing = [c for c in CORE_ACTUAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Actuals are missing required columns: {missing}")
    if df.duplicated(["settlement_date", "region"]).any():
        raise ValueError("Actuals must contain at most one row per settlement_date/region")


def join_actual_tables(
    prices: pd.DataFrame,
    region_summary: pd.DataFrame,
    interconnectors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the canonical actual-data panel from raw ingestion outputs.

    This function only joins source tables; it does not create model features.

    Interconnector data can be either:
      1. already-wide, one row per settlement_date; or
      2. long AEMO DISPATCHINTERCONNECTORRES rows with INTERCONNECTORID.
    """
    p = canonicalise_columns(prices)
    r = canonicalise_columns(region_summary)

    if "intervention" in p.columns:
        p = p[p["intervention"] == 0]
    if "intervention" in r.columns:
        r = r[r["intervention"] == 0]

    price_cols = [c for c in ("settlement_date", "region", "rrp") if c in p.columns]
    p = p[price_cols].drop_duplicates(["settlement_date", "region"], keep="last")
    r = r.drop_duplicates(["settlement_date", "region"], keep="last")

    df = p.merge(r, on=["settlement_date", "region"], how="inner", suffixes=("", "_region"))

    if interconnectors is not None and not interconnectors.empty:
        ic = canonicalise_columns(interconnectors)
        if "intervention" in ic.columns:
            ic = ic[ic["intervention"] == 0]
        ic = _make_interconnectors_wide(ic)
        df = df.merge(ic, on="settlement_date", how="left")

    df = df.sort_values(["region", "settlement_date"]).reset_index(drop=True)
    validate_actuals(df)
    return df


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _make_interconnectors_wide(df: pd.DataFrame) -> pd.DataFrame:
    if "interconnectorid" not in df.columns:
        if df.duplicated("settlement_date").any():
            raise ValueError(
                "Wide interconnector input must have one row per settlement_date"
            )
        return df.drop(columns=["region"], errors="ignore")

    metrics = [c for c in ("mwflow", "importlimit", "exportlimit") if c in df.columns]
    if not metrics:
        return df[["settlement_date"]].drop_duplicates()

    long = (
        df.groupby(["settlement_date", "interconnectorid"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    wide = long.pivot(index="settlement_date", columns="interconnectorid", values=metrics)
    wide.columns = [f"ic_{_slug(ic)}_{metric}" for metric, ic in wide.columns]
    return wide.reset_index()


def add_targets(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add exact future RRP targets without relying on row offsets.

    Exact timestamp joins prevent a missing 5-minute interval from accidentally
    turning a 10-minute move into a nominal 5-minute target.
    """
    out = df.copy()
    lookup = out[["settlement_date", "region", "rrp"]].rename(
        columns={"settlement_date": "target_time", "rrp": "future_rrp"}
    )

    for horizon in config.target_horizons_minutes:
        target_time = out["settlement_date"] + pd.to_timedelta(horizon, unit="m")
        key = pd.DataFrame({"target_time": target_time, "region": out["region"]}, index=out.index)
        matched = key.merge(lookup, on=["target_time", "region"], how="left")
        out[f"target_rrp_h{horizon:02d}"] = matched["future_rrp"].to_numpy()

    if len(config.target_horizons_minutes) == 1:
        h = config.target_horizons_minutes[0]
        out["target_rrp"] = out[f"target_rrp_h{h:02d}"]
    return out


def add_core_engineered_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add the compact, high-value engineered feature set.

    The feature set deliberately focuses on price memory, demand dynamics, supply
    tightness, interconnector tightness, and calendar periodicity. More specialised
    features (constraints, FCAS, bids, weather) can be added by composing another
    function after this one.
    """
    out = df.copy().sort_values(["region", "settlement_date"]).reset_index(drop=True)
    g = out.groupby("region", sort=False)

    # Price memory: immediate, hourly-ish, and same interval previous day.
    for lag in config.price_lags:
        out[f"rrp_lag_{lag}"] = g["rrp"].shift(lag)

    # Demand history is useful without duplicating every possible source variable.
    for lag in config.demand_lags:
        out[f"demand_lag_{lag}"] = g["totaldemand"].shift(lag)

    # Change / ramp features.
    for lag in config.delta_intervals:
        out[f"rrp_delta_{lag}"] = out["rrp"] - g["rrp"].shift(lag)
        out[f"demand_delta_{lag}"] = out["totaldemand"] - g["totaldemand"].shift(lag)

    # Rolling state including the current information-time observation. This is
    # leakage-safe when predicting a future interval because current RRP/demand are
    # known at prediction time.
    for window in config.rolling_windows:
        out[f"rrp_roll_mean_{window}"] = g["rrp"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 3)).mean()
        )
        out[f"rrp_roll_std_{window}"] = g["rrp"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 3)).std()
        )
        out[f"demand_roll_mean_{window}"] = g["totaldemand"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 3)).mean()
        )

    # Supply tightness.
    out["reserve_margin_mw"] = out["availablegeneration"] - out["totaldemand"]
    out["reserve_ratio"] = out["reserve_margin_mw"] / out["totaldemand"].replace(0, np.nan)

    if "demandforecast" in out.columns:
        out["demand_forecast_error_mw"] = out["totaldemand"] - out["demandforecast"]

    if config.include_interconnector_utilisation:
        out = add_interconnector_utilisation(out)

    if config.include_calendar:
        out = add_calendar_features(out)

    if config.clip_infinite_ratios:
        out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_interconnector_utilisation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prefixes = {
        c.removesuffix("_mwflow")
        for c in out.columns
        if c.startswith("ic_") and c.endswith("_mwflow")
    }
    for prefix in sorted(prefixes):
        flow = f"{prefix}_mwflow"
        imp = f"{prefix}_importlimit"
        exp = f"{prefix}_exportlimit"
        if imp not in out.columns or exp not in out.columns:
            continue
        limit = pd.concat([out[imp].abs(), out[exp].abs()], axis=1).max(axis=1)
        out[f"{prefix}_utilisation"] = out[flow].abs() / limit.replace(0, np.nan)
        out[f"{prefix}_headroom_mw"] = limit - out[flow].abs()
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["settlement_date"]
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    out["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    dow = dt.dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype("int8")
    return out


def align_p5min_forecasts(
    actual_index: pd.DataFrame,
    p5min: pd.DataFrame,
    config: FeatureConfig,
    value_columns: Iterable[str] = P5MIN_VALUE_COLUMNS,
) -> pd.DataFrame:
    """Align the latest P5MIN forecasts available at each information time.

    Returns only keys + forecast features so it can be merged onto an engineered
    actual frame. NEMSEER/AEMO RUN_DATETIME labels the run one dispatch interval
    after its effective information time, so availability is treated as
    RUN_DATETIME - interval_minutes. INTERVAL_DATETIME remains the future interval
    being forecast. Keeping these timestamps separate is essential for leakage-safe
    historical experiments.
    """
    base = canonicalise_columns(actual_index)[["settlement_date", "region"]].drop_duplicates()
    fc = canonicalise_columns(p5min)

    required = {"region", "run_datetime", "forecast_time"}
    missing = required - set(fc.columns)
    if missing:
        raise ValueError(f"P5MIN data is missing required columns: {sorted(missing)}")

    fc["available_at"] = (
    pd.to_datetime(fc["run_datetime"])
    - pd.Timedelta(minutes=config.interval_minutes)
    )
    fc["forecast_time"] = pd.to_datetime(fc["forecast_time"])

    values = [c for c in value_columns if c in fc.columns]
    if not values:
        raise ValueError("No requested P5MIN value columns were found")

    result = base.copy()
    for horizon in config.p5min_horizons_minutes:
        left = base.copy()
        left["forecast_time"] = left["settlement_date"] + pd.to_timedelta(horizon, unit="m")
        left = left.sort_values(["settlement_date", "region", "forecast_time"])

        right = fc[["region", "forecast_time", "available_at", *values]].copy()
        right = right.sort_values(["available_at", "region", "forecast_time"])

        aligned = pd.merge_asof(
            left,
            right,
            left_on="settlement_date",
            right_on="available_at",
            by=["region", "forecast_time"],
            direction="backward",
            allow_exact_matches=True,
        )

        rename = {c: f"p5_h{horizon:02d}_{c}" for c in values}
        aligned = aligned[["settlement_date", "region", *values]].rename(columns=rename)
        result = result.merge(aligned, on=["settlement_date", "region"], how="left")

    return result


def add_p5min_forecast_features(
    df: pd.DataFrame,
    p5min: pd.DataFrame | None,
    config: FeatureConfig,
) -> pd.DataFrame:
    if p5min is None or p5min.empty:
        return df
    aligned = align_p5min_forecasts(df, p5min, config)
    out = df.merge(aligned, on=["settlement_date", "region"], how="left")

    # Useful corrections relative to current state; still compact and interpretable.
    for horizon in config.p5min_horizons_minutes:
        p = f"p5_h{horizon:02d}_rrp"
        d = f"p5_h{horizon:02d}_totaldemand"
        a = f"p5_h{horizon:02d}_availablegeneration"
        if p in out.columns:
            out[f"p5_h{horizon:02d}_rrp_vs_now"] = out[p] - out["rrp"]
        if d in out.columns:
            out[f"p5_h{horizon:02d}_demand_vs_now"] = out[d] - out["totaldemand"]
        if d in out.columns and a in out.columns:
            out[f"p5_h{horizon:02d}_reserve_margin_mw"] = out[a] - out[d]
    return out


def add_forecast_reliability_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Add the recent accuracy of AEMO's own P5MIN price forecast.

    For each configured horizon, the forecast issued for the current interval is
    compared with the price that was realised. Rolling windows summarise that
    error (mean and mean absolute), and the longest window also counts the
    forecasts at or above the spike level, the false alarms and the missed
    spikes. A value at time t uses forecasts issued before t and prices up to t,
    so it is known at prediction time. Horizons without an aligned forecast
    column are skipped.
    """
    out = df.copy().sort_values(["region", "settlement_date"]).reset_index(drop=True)
    level = config.reliability_spike_level
    longest = max(config.reliability_windows, default=0)

    def rolling(values: pd.Series, window: int, statistic: str) -> pd.Series:
        return values.groupby(out["region"], sort=False).transform(
            lambda s: getattr(s.rolling(window, min_periods=1), statistic)()
        )

    for minutes in config.reliability_horizons_minutes:
        column = f"p5_h{minutes:02d}_rrp"
        if column not in out.columns:
            continue
        # AEMO's forecast for this interval, issued `minutes` earlier. An exact
        # timestamp join stops a missing interval from shifting the comparison.
        key = pd.DataFrame(
            {
                "issued_at": out["settlement_date"] - pd.to_timedelta(minutes, unit="m"),
                "region": out["region"],
            }
        )
        lookup = out[["settlement_date", "region", column]].rename(
            columns={"settlement_date": "issued_at", column: "issued"}
        )
        matched = key.merge(lookup, on=["issued_at", "region"], how="left")
        issued = pd.Series(matched["issued"].to_numpy(), index=out.index)

        error = out["rrp"] - issued
        for window in config.reliability_windows:
            stem = f"reliability_rrp_err{minutes:02d}"
            out[f"{stem}_mean_{window}"] = rolling(error, window, "mean")
            out[f"{stem}_abs_{window}"] = rolling(error.abs(), window, "mean")

        warned = issued >= level
        counts = {
            "warnings": warned,
            "false_alarms": warned & (out["rrp"] < level),
            "missed": ~warned & issued.notna() & (out["rrp"] >= level),
        }
        for name, flags in counts.items():
            out[f"reliability_{name}{minutes:02d}_{longest}"] = rolling(
                flags.astype(float), longest, "sum"
            )
    return out


def forecast_reliability_columns(df: pd.DataFrame) -> list[str]:
    """Return the forecast reliability features present in ``df``."""
    return [c for c in df.columns if c.startswith("reliability_")]


def build_features(
    actuals: pd.DataFrame,
    *,
    p5min: pd.DataFrame | None = None,
    p5min_history: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    config: FeatureConfig | None = None,
    add_target_columns: bool = True,
) -> pd.DataFrame:
    """Build features for one split, optionally using preceding history for lags.

    Pass the training actual frame as ``history`` when transforming validation/test
    data. If forecast data is split by run time, pass the tail of the training P5MIN
    data as ``p5min_history`` so the first test intervals can use forecasts that were
    published just before the split. Only rows belonging to ``actuals`` are returned.
    """
    config = config or FeatureConfig()
    current = canonicalise_columns(actuals)
    validate_actuals(current)

    current_keys = current[["settlement_date", "region"]].copy()

    if history is not None and not history.empty:
        hist = canonicalise_columns(history)
        validate_actuals(hist)
        # Only the required tail is needed for lag/rolling context.
        hist = (
            hist.sort_values(["region", "settlement_date"])
            .groupby("region", group_keys=False)
            .tail(config.max_history_intervals)
        )
        combined = pd.concat([hist, current], ignore_index=True)
        combined = combined.drop_duplicates(["settlement_date", "region"], keep="last")
    else:
        combined = current.copy()

    combined = add_core_engineered_features(combined, config)

    forecast_input = p5min
    if p5min_history is not None and not p5min_history.empty:
        if forecast_input is None or forecast_input.empty:
            forecast_input = p5min_history
        else:
            forecast_input = pd.concat([p5min_history, forecast_input], ignore_index=True)
            keys = [c for c in ("REGIONID", "region", "RUN_DATETIME", "run_datetime", "INTERVAL_DATETIME", "forecast_time") if c in forecast_input.columns]
            if keys:
                forecast_input = forecast_input.drop_duplicates(keys, keep="last")

    combined = add_p5min_forecast_features(combined, forecast_input, config)
    combined = add_forecast_reliability_features(combined, config)
    if add_target_columns:
        combined = add_targets(combined, config)

    out = current_keys.merge(combined, on=["settlement_date", "region"], how="left")
    return out.sort_values(["settlement_date", "region"]).reset_index(drop=True)


def essential_feature_columns(df: pd.DataFrame, config: FeatureConfig | None = None) -> list[str]:
    """Return the deliberately constrained v1 feature set present in ``df``.

    New feature families can be appended elsewhere without silently changing the
    baseline model. This makes ablation tests and comparisons reproducible. The
    forecast reliability features are one such family: they are built for every
    row but only become model inputs when an experiment asks for them.
    """
    config = config or FeatureConfig()
    wanted: list[str] = []

    for c in (
        "rrp",
        "totaldemand",
        "availablegeneration",
        "demandforecast",
        "dispatchablegeneration",
        "dispatchableload",
        "netinterchange",
        "reserve_margin_mw",
        "reserve_ratio",
        "demand_forecast_error_mw",
        "time_sin",
        "time_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ):
        if c in df.columns:
            wanted.append(c)

    for lag in config.price_lags:
        c = f"rrp_lag_{lag}"
        if c in df.columns:
            wanted.append(c)
    for lag in config.demand_lags:
        c = f"demand_lag_{lag}"
        if c in df.columns:
            wanted.append(c)
    for lag in config.delta_intervals:
        for stem in ("rrp_delta", "demand_delta"):
            c = f"{stem}_{lag}"
            if c in df.columns:
                wanted.append(c)
    for window in config.rolling_windows:
        for stem in ("rrp_roll_mean", "rrp_roll_std", "demand_roll_mean"):
            c = f"{stem}_{window}"
            if c in df.columns:
                wanted.append(c)

    # Interconnector engineered features only, plus raw flow/limits so trees can
    # learn direction-specific effects.
    wanted.extend(
        c for c in df.columns
        if c.startswith("ic_") and c.endswith(("_mwflow", "_importlimit", "_exportlimit", "_utilisation", "_headroom_mw"))
    )

    # Known-future AEMO trajectory.
    wanted.extend(c for c in df.columns if c.startswith("p5_h"))

    return list(dict.fromkeys(wanted))
