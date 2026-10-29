"""Node-aligned tensors built from the all-region prepared feature frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.contracts import validate_timeline

from .config import DEFAULT_NODE_FEATURES, GraphEmbeddingConfig
from .graph import EDGE_FEATURES, edge_pairs


@dataclass
class GraphPanel:
    """Features and targets on one complete time axis shared by every region.

    x: (time, num_nodes, num_features); y: (time, num_nodes) on the experiment's
    target scale. Intervals a region did not report are NaN, never dropped, so
    row offsets always correspond to elapsed time.

    future_price: (time, num_nodes) RRP at the forecast horizon.
    edges: (time, num_nodes, num_nodes, len(EDGE_FEATURES)) interconnector
    state, zero between unlinked regions; None without interconnector data.
    Flow is signed, so ``edges[t, a, b]`` and ``edges[t, b, a]`` are opposite.
    """

    times: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    regions: tuple[str, ...]
    node_features: tuple[str, ...]
    future_price: np.ndarray | None = None
    edges: np.ndarray | None = None


def resolve_node_features(
    frame: pd.DataFrame,
    config: GraphEmbeddingConfig,
    horizon: int,
) -> tuple[str, ...]:
    """Choose configured node inputs that exist and carry data in ``frame``."""
    if config.node_features is not None:
        missing = [name for name in config.node_features if name not in frame.columns]
        if missing:
            raise ValueError(f"Graph node features are missing: {missing}")
        return tuple(dict.fromkeys(config.node_features))
    wanted = [*DEFAULT_NODE_FEATURES, f"p5_h{horizon:02d}_rrp_vs_now"]
    selected = tuple(
        name for name in wanted if name in frame.columns and frame[name].notna().any()
    )
    if not selected:
        raise ValueError("No graph node features are available in the prepared data")
    return selected


def interconnector_columns(regions: Sequence[str]) -> list[str]:
    """Every interconnector column the edge tensor can use for these regions."""
    return [
        f"{prefix}_{suffix}"
        for _, _, prefixes in edge_pairs(regions)
        for prefix in prefixes
        for suffix in ("mwflow", "utilisation", "headroom_mw")
    ]


def build_edges(
    frame: pd.DataFrame, regions: Sequence[str], times: pd.DatetimeIndex
) -> np.ndarray | None:
    """Aggregate parallel links into one flow, utilisation and headroom per pair."""
    available = [c for c in interconnector_columns(regions) if c in frame.columns]
    if not any(column.endswith("_mwflow") for column in available):
        return None
    # Interconnector columns repeat on every region's row for a timestamp.
    links = (
        frame.drop_duplicates("settlement_date")
        .set_index("settlement_date")[available]
        .apply(pd.to_numeric, errors="coerce")
        .reindex(times)
    )
    edges = np.zeros(
        (len(times), len(regions), len(regions), len(EDGE_FEATURES)), dtype=np.float32
    )
    for first, second, prefixes in edge_pairs(regions):

        def metric(suffix: str) -> pd.DataFrame:
            return links[[f"{p}_{suffix}" for p in prefixes if f"{p}_{suffix}" in links]]

        flow = metric("mwflow").sum(axis=1, min_count=1).to_numpy(dtype=np.float32)
        utilisation = metric("utilisation").max(axis=1).to_numpy(dtype=np.float32)
        headroom = (
            metric("headroom_mw").sum(axis=1, min_count=1).to_numpy(dtype=np.float32)
        )
        edges[:, first, second] = np.stack([flow, utilisation, headroom], axis=1)
        edges[:, second, first] = np.stack([-flow, utilisation, headroom], axis=1)
    edges[np.isinf(edges)] = np.nan
    return edges


def build_graph_panel(
    frame: pd.DataFrame,
    regions: Sequence[str],
    node_features: Sequence[str],
    *,
    interval_minutes: int = 5,
    edge_features: bool = True,
) -> GraphPanel:
    """Pivot a long (region, settlement_date) frame that has a ``target`` column."""
    present = set(frame["region"].unique())
    absent = [region for region in regions if region not in present]
    if absent:
        raise ValueError(f"Graph regions are missing from the prepared data: {absent}")

    moments = pd.to_datetime(frame["settlement_date"])
    times = pd.date_range(
        moments.min(), moments.max(), freq=pd.Timedelta(minutes=interval_minutes)
    )
    validate_timeline(
        pd.DataFrame({"region": "graph", "settlement_date": times}), interval_minutes
    )

    has_future = "target_rrp" in frame.columns
    columns = [*node_features, "target", *(["target_rrp"] if has_future else [])]
    x = np.full((len(times), len(regions), len(node_features)), np.nan, dtype=np.float32)
    y = np.full((len(times), len(regions)), np.nan, dtype=np.float32)
    future = np.full((len(times), len(regions)), np.nan, dtype=np.float32)
    for position, region in enumerate(regions):
        node = frame.loc[frame["region"] == region].set_index("settlement_date")
        if node.index.has_duplicates:
            raise ValueError(f"Region {region} has duplicate timestamps")
        values = (
            node[columns]
            .apply(pd.to_numeric, errors="coerce")
            .reindex(times)
            .to_numpy(dtype=np.float32)
        )
        x[:, position, :] = values[:, : len(node_features)]
        y[:, position] = values[:, len(node_features)]
        if has_future:
            future[:, position] = values[:, -1]
    for array in (x, y, future):
        array[~np.isfinite(array)] = np.nan
    return GraphPanel(
        times,
        x,
        y,
        tuple(regions),
        tuple(node_features),
        future_price=future if has_future else None,
        edges=build_edges(frame, regions, times) if edge_features else None,
    )
