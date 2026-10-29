"""Fit the graph encoder on an early training window and embed every interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from electricity_price_forecasting_model.layers.gnn_layers.encoder import (
    SpatioTemporalGraphEncoder,
)
from pipeline.neural import resolve_device, set_random_seed
from pipeline.preprocessing.adapters import LearnedFeatures, add_model_target
from pipeline.preprocessing.scaling import (
    NumericPreprocessor,
    TargetTransformer,
    suggested_scaling_groups,
)
from pipeline.sequences import ContextSequenceDataset, SequenceDataset
from pipeline.training import TorchEarlyStoppingTrainer

from .config import GraphEmbeddingConfig
from .graph import EDGE_FEATURES, adjacency, edge_pairs
from .panel import (
    GraphPanel,
    build_graph_panel,
    interconnector_columns,
    resolve_node_features,
)


class GraphEmbedder:
    """Supervised encoder whose per-region hidden state becomes model features.

    The encoder learns to forecast every region's target from a window of all
    regions' features. Only labels that resolve before ``fit_end`` are used,
    and the weights are frozen afterwards, so embeddings for later intervals
    are computed by a model that never saw their outcomes.

    Two optional parts make the embedding carry cross-region state rather
    than a second copy of the region's own forecast: interconnector state
    enters the attention scores, and the encoder is also trained to forecast
    the price difference across each interconnector.
    """

    def __init__(self, config: GraphEmbeddingConfig, *, interval_minutes: int = 5):
        self.config = config
        self.interval_minutes = interval_minutes
        self.device = resolve_device(config.device)
        self.network = None
        self.regions: tuple[str, ...] = ()
        self.node_features: tuple[str, ...] = ()
        self.feature_medians: list[pd.Series] = []
        self.preprocessors: list[NumericPreprocessor] = []
        self.target_transformers: list[TargetTransformer] = []
        self.spread_transformers: list[TargetTransformer] = []
        self.pairs: list[tuple[int, int]] = []
        self.edge_statistics: dict[str, float] | None = None
        self.edge_features_used: list[str] = []
        self.fit_end = None
        self.diagnostics: dict = {}

    def _scale(self, x: np.ndarray, edges: np.ndarray | None = None) -> torch.Tensor:
        """Impute and scale each node with its own fitted statistics."""
        scaled = np.empty_like(x, dtype=np.float32)
        for node, (medians, preprocessor) in enumerate(
            zip(self.feature_medians, self.preprocessors)
        ):
            frame = pd.DataFrame(x[:, node, :], columns=list(self.node_features))
            scaled[:, node, :] = preprocessor.transform(frame.fillna(medians)).to_numpy(
                dtype=np.float32
            )
        if self.edge_statistics is None:
            return torch.from_numpy(scaled)
        return torch.from_numpy(
            np.concatenate([scaled, self._scale_edges(edges)], axis=-1)
        )

    def _scale_edges(self, edges: np.ndarray) -> np.ndarray:
        """Scale edge attributes and pack each node's row beside its features.

        Flow is divided by its spread but not centred, so the two directions
        of a link stay exact opposites. Missing values become zero.
        """
        statistics = self.edge_statistics
        scaled = edges.copy()
        scaled[..., 0] /= statistics["flow_std"]
        scaled[..., 2] = (scaled[..., 2] - statistics["headroom_mean"]) / statistics[
            "headroom_std"
        ]
        scaled[:, ~self._linked] = 0.0
        return np.nan_to_num(scaled).reshape(len(edges), edges.shape[1], -1)

    @staticmethod
    def _spreads(panel: GraphPanel, pairs) -> np.ndarray:
        """Future price of the first node minus the second, per linked pair."""
        return np.stack(
            [panel.future_price[:, a] - panel.future_price[:, b] for a, b in pairs],
            axis=1,
        )

    def _forward(self, history: torch.Tensor, inputs: torch.Tensor):
        """Return embeddings and transformed node and spread forecasts per row."""
        loader = DataLoader(
            ContextSequenceDataset(history, inputs, self.config.sequence_length),
            batch_size=self.config.prediction_batch_size,
            shuffle=False,
        )
        self.network.eval()
        outputs = []
        with torch.inference_mode():
            for (xb,) in loader:
                outputs.append([o.cpu().numpy() for o in self.network(xb.to(self.device))])
        return tuple(np.concatenate(parts) for parts in zip(*outputs))

    def fit(self, panel: GraphPanel, horizon_minutes: int, train_end=None):
        """Fit on the first ``fit_fraction`` of the intervals before ``train_end``.

        The final ``inner_validation_fraction`` of that window selects the
        checkpoint. Labels within one horizon of either boundary are masked.
        """
        config = self.config
        length = config.sequence_length
        times = panel.times
        horizon = pd.Timedelta(minutes=horizon_minutes)
        available = len(times) if train_end is None else int((times < train_end).sum())
        fit_index = int(available * config.fit_fraction)
        if fit_index <= length:
            raise ValueError("Too few training intervals to fit the graph encoder")
        fit_end = times[fit_index]

        resolved = (times + horizon < fit_end) & np.isfinite(panel.y).any(axis=1)
        eligible = np.flatnonzero(resolved)
        if len(eligible) < 2:
            raise ValueError("Graph encoder fit window contains too few targets")
        inner_start = times[
            eligible[int(len(eligible) * (1 - config.inner_validation_fraction))]
        ]
        train_rows = resolved & (times + horizon < inner_start)
        validation_rows = np.flatnonzero(resolved & (times >= inner_start))
        if not train_rows.any() or len(validation_rows) == 0:
            raise ValueError("Graph encoder split produced an empty subset")

        self.regions, self.node_features = panel.regions, panel.node_features
        nodes = len(panel.regions)
        links = edge_pairs(panel.regions)
        self._linked = np.zeros((nodes, nodes), dtype=bool)
        for a, b, _ in links:
            self._linked[a, b] = self._linked[b, a] = True
        self.edge_statistics = None
        self.edge_features_used = []
        if (
            config.edge_features
            and panel.edges is not None
        ):
            observed = panel.edges[train_rows][:, self._linked]
            # Ingestion may supply flows without limits, leaving utilisation
            # and headroom empty. Those stay in the tensor as zeros.
            self.edge_features_used = [
                name
                for index, name in enumerate(EDGE_FEATURES)
                if np.isfinite(observed[..., index]).any()
            ]
            if self.edge_features_used:
                flow = observed[..., 0][np.isfinite(observed[..., 0])]
                headroom = observed[..., 2][np.isfinite(observed[..., 2])]
                self.edge_statistics = {
                    "flow_std": float(flow.std()) if len(flow) and flow.std() else 1.0,
                    "headroom_mean": float(headroom.mean()) if len(headroom) else 0.0,
                    "headroom_std": float(headroom.std())
                    if len(headroom) and headroom.std()
                    else 1.0,
                }
        self.pairs = (
            [(a, b) for a, b, _ in links]
            if config.spread_loss_weight > 0 and panel.future_price is not None
            else []
        )
        groups = suggested_scaling_groups(list(panel.node_features))
        self.feature_medians, self.preprocessors, self.target_transformers = [], [], []
        targets = np.full((len(times), nodes + len(self.pairs)), np.nan, dtype=np.float32)
        for node, region in enumerate(panel.regions):
            fitting = pd.DataFrame(
                panel.x[train_rows, node, :], columns=list(panel.node_features)
            )
            medians = fitting.median().fillna(0.0)
            self.feature_medians.append(medians)
            self.preprocessors.append(
                NumericPreprocessor(
                    asinh_columns=tuple(groups["asinh_then_standardise"]),
                    standardise_columns=tuple(groups["standardise_only"]),
                ).fit(fitting.fillna(medians))
            )
            labels = panel.y[train_rows, node]
            if not np.isfinite(labels).any():
                raise ValueError(f"Region {region} has no targets in the encoder fit window")
            transformer = TargetTransformer().fit(labels)
            self.target_transformers.append(transformer)
            targets[:, node] = transformer.transform(panel.y[:, node])
        self.spread_transformers = []
        spreads = self._spreads(panel, self.pairs) if self.pairs else None
        for index in range(len(self.pairs)):
            labels = spreads[train_rows, index]
            if not np.isfinite(labels).any():
                raise ValueError("A linked pair has no spreads in the encoder fit window")
            transformer = TargetTransformer().fit(labels)
            self.spread_transformers.append(transformer)
            targets[:, nodes + index] = transformer.transform(spreads[:, index])

        inputs = self._scale(panel.x, panel.edges)
        target_tensor = torch.from_numpy(targets)
        dataset = SequenceDataset(inputs, target_tensor, length, target_mask=train_rows)
        validation_start, validation_stop = validation_rows[0], validation_rows[-1] + 1
        validation_history = inputs[:validation_start]
        validation_inputs = inputs[validation_start:validation_stop]
        validation_targets = targets[validation_start:validation_stop, :nodes]
        # Rows between the two boundaries are embargoed, not scored.
        scored_rows = np.zeros(len(validation_inputs), dtype=bool)
        scored_rows[validation_rows - validation_start] = True
        scored = scored_rows[:, None] & np.isfinite(validation_targets)

        set_random_seed(config.random_seed)
        self.network = SpatioTemporalGraphEncoder(
            len(panel.node_features),
            torch.tensor(adjacency(panel.regions)),
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            edge_size=len(EDGE_FEATURES) if self.edge_statistics else 0,
            spread_pairs=self.pairs,
        ).to(self.device)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.random_seed),
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.HuberLoss(delta=config.huber_delta)

        def train_epoch() -> float:
            self.network.train()
            total, count = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                node_targets, spread_targets = yb[:, :nodes], yb[:, nodes:]
                mask = torch.isfinite(node_targets)
                optimizer.zero_grad(set_to_none=True)
                _, prediction, spread = self.network(xb)
                loss = criterion(prediction[mask], node_targets[mask])
                spread_mask = torch.isfinite(spread_targets)
                if spread_mask.any():
                    loss = loss + config.spread_loss_weight * criterion(
                        spread[spread_mask], spread_targets[spread_mask]
                    )
                loss.backward()
                optimizer.step()
                total += loss.item() * int(mask.sum())
                count += int(mask.sum())
            return total / count

        # Checkpoints are chosen on the node targets alone: that is the signal
        # downstream models need, and it keeps runs comparable across settings.
        def validation_score() -> float:
            _, prediction, _ = self._forward(validation_history, validation_inputs)
            return float(np.mean(np.abs(prediction[scored] - validation_targets[scored])))

        result = TorchEarlyStoppingTrainer(
            max_epochs=config.max_epochs,
            patience=config.patience,
            min_delta=config.min_delta,
        ).fit(
            network=self.network,
            train_epoch=train_epoch,
            validation_score=validation_score,
        )

        _, prediction, spread = self._forward(validation_history, validation_inputs)
        actual = panel.y[validation_start:validation_stop]
        validation_mae, zero_forecast_mae = {}, {}
        for node, region in enumerate(panel.regions):
            rows = scored[:, node]
            forecast = self.target_transformers[node].inverse_transform(prediction[rows, node])
            validation_mae[region] = float(np.mean(np.abs(actual[rows, node] - forecast)))
            zero_forecast_mae[region] = float(np.mean(np.abs(actual[rows, node])))

        spread_mae, spread_zero_mae = {}, {}
        for index, (a, b) in enumerate(self.pairs):
            name = f"{panel.regions[a]}-{panel.regions[b]}"
            observed = spreads[validation_start:validation_stop, index]
            rows = scored_rows & np.isfinite(observed)
            forecast = self.spread_transformers[index].inverse_transform(
                spread[rows, index]
            )
            spread_mae[name] = float(np.mean(np.abs(observed[rows] - forecast)))
            spread_zero_mae[name] = float(np.mean(np.abs(observed[rows])))

        self.fit_end = fit_end
        self.diagnostics = {
            "adjacency_mode": config.adjacency_mode,
            "edge_features": self.edge_features_used,
            "spread_loss_weight": config.spread_loss_weight if self.pairs else 0.0,
            # Future price difference across each interconnector, in $/MWh.
            "spread_validation_mae": spread_mae,
            "spread_zero_forecast_mae": spread_zero_mae,
            "node_features": list(panel.node_features),
            "fit_end": str(fit_end),
            "inner_validation_start": str(inner_start),
            "training_windows": len(dataset),
            "best_epoch": result.best_epoch,
            "stopped_epoch": result.stopped_epoch,
            "training_loss_history": result.training_loss_history,
            "validation_score_history": result.validation_score_history,
            # Encoder head on its inner validation slice, in target units. A
            # zero forecast equals persistence when the target is ``change``.
            "validation_mae": validation_mae,
            "zero_forecast_mae": zero_forecast_mae,
        }
        return self

    def transform(self, panel: GraphPanel) -> pd.DataFrame:
        """Return ``gnn_emb_*`` columns per region and interval, from past rows only.

        The first ``sequence_length - 1`` intervals lack context and are NaN.
        """
        if self.network is None:
            raise RuntimeError("fit() must be called before transform()")
        if panel.regions != self.regions or panel.node_features != self.node_features:
            raise ValueError("Panel regions and features must match the fitted encoder")
        columns = self.config.embedding_columns
        count = self.config.sequence_length - 1
        embedding = np.full(
            (len(panel.times), len(panel.regions), len(columns)), np.nan, dtype=np.float32
        )
        if len(panel.times) > count:
            if self.edge_statistics is not None and panel.edges is None:
                raise ValueError("Panel lacks the edge attributes the encoder was fitted on")
            inputs = self._scale(panel.x, panel.edges)
            embedding[count:] = self._forward(inputs[:count], inputs[count:])[0]
        frames = []
        for node, region in enumerate(panel.regions):
            frame = pd.DataFrame(embedding[:, node, :], columns=columns)
            frame.insert(0, "settlement_date", panel.times)
            frame.insert(0, "region", region)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def compute_graph_embeddings(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    config: GraphEmbeddingConfig,
    horizon: int,
    target: str,
    interval_minutes: int = 5,
) -> LearnedFeatures:
    """Fit the encoder on early training data and embed both splits.

    Both splits share one time axis so the first test windows continue from
    the training tail. Fitting still sees training-period labels only.
    """
    train = add_model_target(train_features, target, horizon, split="training")
    test = add_model_target(test_features, target, horizon, split="test")
    node_features = resolve_node_features(train, config, horizon)
    absent = [name for name in node_features if name not in test.columns]
    if absent:
        raise ValueError(f"Graph node features are missing from test data: {absent}")
    links = [
        name
        for name in interconnector_columns(config.regions)
        if name in train.columns and name in test.columns
    ]
    keep = list(
        dict.fromkeys(
            ["region", "settlement_date", *node_features, "target", "target_rrp", *links]
        )
    )
    combined = pd.concat([train[keep], test[keep]], ignore_index=True)
    panel = build_graph_panel(
        combined,
        config.regions,
        node_features,
        interval_minutes=interval_minutes,
        edge_features=config.edge_features,
    )
    train_end = pd.to_datetime(test["settlement_date"]).min()

    embedder = GraphEmbedder(config, interval_minutes=interval_minutes)
    embedder.fit(panel, horizon, train_end=train_end)
    embedded = embedder.transform(panel)
    is_test = embedded["settlement_date"] >= train_end
    return LearnedFeatures(
        train=embedded.loc[~is_test].reset_index(drop=True),
        test=embedded.loc[is_test].reset_index(drop=True),
        columns=config.embedding_columns,
        fit_end=embedder.fit_end,
        diagnostics=embedder.diagnostics,
    )
