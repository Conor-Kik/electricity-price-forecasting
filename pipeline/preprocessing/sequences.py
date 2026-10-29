from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.contracts import validate_timeline


@dataclass
class SequenceBatch:
    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    regions: np.ndarray
    feature_columns: list[str]
    target_columns: list[str]


def make_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_columns: list[str] | None = None,
    sequence_length: int = 288,
    history: pd.DataFrame | None = None,
    interval_minutes: int = 5,
    drop_nan: bool = True,
    dtype: np.dtype = np.float32,
) -> SequenceBatch:
    """Convert a feature frame into model-agnostic sequence arrays.

    Pass the preceding split as ``history`` when building validation/test
    sequences. Its final feature rows provide context, while targets and
    metadata are always taken from ``df``.

    This eager helper is useful for analysis and export. PyTorch models should
    generally use the lazy datasets in :mod:`pipeline.sequences` to avoid
    materialising overlapping windows.
    """
    if sequence_length <= 0 or int(sequence_length) != sequence_length:
        raise ValueError("sequence_length must be a positive integer")
    sequence_length = int(sequence_length)
    target_columns = target_columns or ["target_rrp"]
    required = ["region", "settlement_date", *feature_columns, *target_columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sequence columns: {missing}")
    if history is not None:
        history_required = ["region", "settlement_date", *feature_columns]
        history_missing = [c for c in history_required if c not in history.columns]
        if history_missing:
            raise ValueError(f"Missing history sequence columns: {history_missing}")

    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    times: list[np.datetime64] = []
    regions: list[str] = []

    for region, part in df.groupby("region", sort=False):
        part = part.reset_index(drop=True)
        validate_timeline(part, interval_minutes)
        current_x = part[feature_columns].to_numpy(dtype=dtype)
        y = part[target_columns].to_numpy(dtype=dtype)
        t = part["settlement_date"].to_numpy()

        context = np.empty((0, len(feature_columns)), dtype=dtype)
        if history is not None and sequence_length > 1:
            history_part = history.loc[history["region"] == region]
            validate_timeline(history_part, interval_minutes)
            if (
                not history_part.empty
                and history_part["settlement_date"].max()
                >= part["settlement_date"].min()
            ):
                raise ValueError("History must strictly precede prediction inputs")
            validate_timeline(
                pd.concat(
                    [history_part.tail(sequence_length - 1), part], ignore_index=True
                ),
                interval_minutes,
            )
            context = (
                history_part[feature_columns]
                .tail(sequence_length - 1)
                .to_numpy(dtype=dtype)
            )

        x = np.concatenate((context, current_x), axis=0)
        context_length = len(context)
        for end in range(max(context_length, sequence_length - 1), len(x)):
            start = end - sequence_length + 1
            x_window = x[start : end + 1]
            current_index = end - context_length
            y_value = y[current_index]
            if drop_nan and (
                not np.isfinite(x_window).all() or not np.isfinite(y_value).all()
            ):
                continue
            Xs.append(x_window)
            ys.append(y_value)
            times.append(t[current_index])
            regions.append(str(region))

    n_features = len(feature_columns)
    n_targets = len(target_columns)
    X = np.stack(Xs) if Xs else np.empty((0, sequence_length, n_features), dtype=dtype)
    y = np.stack(ys) if ys else np.empty((0, n_targets), dtype=dtype)

    return SequenceBatch(
        X=X,
        y=y,
        timestamps=np.asarray(times),
        regions=np.asarray(regions),
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
