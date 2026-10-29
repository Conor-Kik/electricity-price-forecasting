"""Fixed NEM topology: regions are nodes, interconnectors are edges."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_REGIONS: tuple[str, ...] = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")

# Keys are the ``ic_<id>`` column prefixes produced by preprocessing, so edge
# attributes can later be looked up from the same table.
NEM_INTERCONNECTORS: dict[str, tuple[str, str]] = {
    "ic_nsw1_qld1": ("NSW1", "QLD1"),  # QNI
    "ic_n_q_mnsp1": ("NSW1", "QLD1"),  # Terranora
    "ic_vic1_nsw1": ("VIC1", "NSW1"),  # VNI
    "ic_v_sa": ("VIC1", "SA1"),  # Heywood
    "ic_v_s_mnsp1": ("VIC1", "SA1"),  # Murraylink
    "ic_t_v_mnsp1": ("TAS1", "VIC1"),  # Basslink
}


EDGE_FEATURES: tuple[str, ...] = ("flow", "utilisation", "headroom")


def edge_pairs(
    regions: Sequence[str] = DEFAULT_REGIONS,
) -> list[tuple[int, int, tuple[str, ...]]]:
    """Return ``(from_node, to_node, column_prefixes)`` once per linked pair.

    Parallel interconnectors share one pair. The orientation follows the
    table, which is the direction of positive metered flow.
    """
    position = {region: index for index, region in enumerate(regions)}
    grouped: dict[tuple[int, int], list[str]] = {}
    for prefix, (first, second) in NEM_INTERCONNECTORS.items():
        if first in position and second in position:
            grouped.setdefault((position[first], position[second]), []).append(prefix)
    return [(a, b, tuple(prefixes)) for (a, b), prefixes in grouped.items()]


def adjacency(regions: Sequence[str] = DEFAULT_REGIONS) -> np.ndarray:
    """Return the NEM interconnector matrix with self-loops in the given order."""
    position = {region: index for index, region in enumerate(regions)}
    if len(position) != len(regions):
        raise ValueError("regions must be distinct")
    matrix = np.eye(len(regions), dtype=np.float32)
    for first, second in NEM_INTERCONNECTORS.values():
        if first in position and second in position:
            matrix[position[first], position[second]] = 1.0
            matrix[position[second], position[first]] = 1.0
    return matrix
