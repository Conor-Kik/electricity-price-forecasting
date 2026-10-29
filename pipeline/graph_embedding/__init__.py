"""Learned cross-region embeddings appended to the prepared model features.

The encoder itself (``embedder``) imports torch and is loaded lazily by the
experiment pipeline, so importing this package stays cheap.
"""

from .config import DEFAULT_NODE_FEATURES, GraphEmbeddingConfig
from .graph import (
    DEFAULT_REGIONS,
    EDGE_FEATURES,
    NEM_INTERCONNECTORS,
    adjacency,
    edge_pairs,
)

__all__ = [
    "DEFAULT_NODE_FEATURES",
    "DEFAULT_REGIONS",
    "EDGE_FEATURES",
    "GraphEmbeddingConfig",
    "NEM_INTERCONNECTORS",
    "adjacency",
    "edge_pairs",
]
