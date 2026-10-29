"""Shared preprocessing, training and inference for neural forecasting models."""

from __future__ import annotations

import random
import copy
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from collections.abc import Iterable, Sequence
from contextlib import ExitStack

from tqdm.auto import tqdm

from pipeline.contracts import (
    METRIC_NAMES,
    forecast_array,
    score_forecast,
    temporal_metadata,
    validate_context,
    validate_timeline,
)
from pipeline.model import Model
from pipeline.preprocessing import (
    NumericPreprocessor,
    TargetTransformer,
    suggested_scaling_groups,
)
from pipeline.sequences import ContextSequenceDataset, SequenceDataset
from pipeline.training import TorchEarlyStoppingTrainer


def resolve_device(device=None) -> torch.device:
    """Use an explicit device or prefer an available CUDA/Intel GPU."""
    if device is None or str(device) == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    return torch.device(device)


def set_random_seed(seed: int) -> None:
    """Seed every generator a PyTorch fit depends on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TorchForecastModel(Model):
    """Subclasses supply a network; this wrapper owns the forecasting lifecycle.

    Sequence inputs use a DatetimeIndex or a (region, settlement_date) index.
    Unindexed frames remain supported for positional experiments, but cannot
    receive timestamp checks. Production adapters always supply metadata.

    Subclasses define architecture parameters and implement ``_build_network``.
    Row networks receive (batch, features); sequence networks receive
    (batch, lookback, features). DataFrame targets configure multiple outputs;
    Series targets retain the original single-output behavior.
    Public ``predict`` converts these back to the experiment's target units.
    """

    sequence_length = 1
    output_size = 1
    show_progress: bool = True

    def __init__(
        self,
        *,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        prediction_batch_size=1024,
        max_epochs=30,
        patience=3,
        min_delta=0.001,
        loss="huber",
        huber_delta=1.0,
        random_seed=42,
        device=None,
        selection_metric="mae",
        gradient_clip_norm=None,
        scheduler_step_size=None,
        scheduler_gamma=0.5,
        mixed_precision=False,
        interval_minutes=5,
    ):
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if min(batch_size, prediction_batch_size, max_epochs, patience) <= 0:
            raise ValueError("batch sizes, max_epochs and patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        loss = loss.lower()
        if loss not in {"huber", "mse", "mae"}:
            raise ValueError("loss must be 'huber', 'mse', or 'mae'")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if selection_metric not in METRIC_NAMES:
            raise ValueError(f"selection_metric must be one of {METRIC_NAMES}")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if scheduler_step_size is not None and scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be positive")
        if not 0 < scheduler_gamma <= 1 or interval_minutes <= 0:
            raise ValueError("Invalid scheduler_gamma or interval_minutes")
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.prediction_batch_size = int(prediction_batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.loss = loss
        self.huber_delta = float(huber_delta)
        self.random_seed = int(random_seed)
        self.selection_metric = selection_metric
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.mixed_precision = mixed_precision
        self.interval_minutes = interval_minutes
        self.device = resolve_device(device)
        self.network = None
        self.preprocessor = None
        self.target_transformer = None
        self.feature_columns = []
        self.target_columns = None
        self.feature_medians = None
        self.training_context = None
        self.context_frame = None
        self.training_loss_history = []
        self.validation_score_history = []
        self.validation_mae_history = []
        self.best_epoch = None
        self.stopped_epoch = None
        self.training_result = None

    @abstractmethod
    def _build_network(self, input_size: int) -> nn.Module:
        """Return an untrained network with self.output_size outputs per sample."""

    def _forecast_values(self, values, expected_rows):
        """Preserve the horizon axis while keeping single-output predictions 1D."""
        return forecast_array(values, expected_rows, self.output_size)

    def _set_random_seed(self):
        set_random_seed(self.random_seed)

    def _build_criterion(self):
        """Choose the optimization loss, evaluated in transformed target space."""
        if self.loss == "mse":
            return nn.MSELoss()
        if self.loss == "mae":
            return nn.L1Loss()
        return nn.HuberLoss(delta=self.huber_delta)

    @staticmethod
    def _numeric_frame(frame):
        return frame.apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    def _impute_features(self, frame):
        """Restore training column order and fill missing values with fitted medians."""
        if self.feature_medians is None:
            raise RuntimeError("train() must be called before predict()")
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing model features: {', '.join(sorted(missing))}")
        return self._numeric_frame(frame[self.feature_columns]).fillna(
            self.feature_medians
        )

    def _inputs(self, frame):
        """Apply existing transforms and create float32 CPU tensors; never refit here."""
        scaled = self.preprocessor.transform(self._impute_features(frame))
        return torch.tensor(scaled.to_numpy(dtype=np.float32))

    def _check_timeline(self, frame):
        metadata = temporal_metadata(frame)
        if metadata is not None:
            validate_timeline(metadata, self.interval_minutes)

    def gradient_shap(self, features, reference_features, *, history=None, positions=None,
                      sample_size=1024, baseline_size=16, n_samples=128, device=None, target=None):
        """Signed learned-target contributions relative to sampled training inputs.

        Uses complete observed windows for sequence references and no added noise.
        Defaults to the model's GPU, then available CUDA/XPU, then CPU. Pass
        device="cpu" to force CPU attribution. GPUs use float32 and chunked
        draws; CPU retains float64. The fitted network is never modified.
        The residual compares attribution sums with prediction minus the mean
        reference prediction; Monte Carlo approximation need not sum exactly.
        Multi-output models require target: the zero-based output column index.
        """
        if self.network is None or n_samples < 1 or baseline_size < 1 or features.empty or (sample_size is not None and sample_size < 1):
            raise ValueError("A fitted model, nonempty inputs and positive integration settings are required")
        if target is None and self.output_size > 1:
            raise ValueError("Multi-output GradientShap requires a target output index")
        if target is not None and (not isinstance(target, (int, np.integer)) or not 0 <= target < self.output_size):
            raise ValueError(f"target must be an output index from 0 to {self.output_size - 1}")
        length = self.sequence_length if self.requires_sequence else 1
        if len(reference_features) < length:
            raise ValueError("Training references must contain at least one complete input window")
        rng = np.random.default_rng(self.random_seed)
        candidates = np.arange(length - 1, len(reference_features))
        reference_positions = rng.choice(candidates, min(baseline_size, len(candidates)), replace=False)
        if self.requires_sequence:
            self._check_timeline(reference_features)
            reference_inputs = self._inputs(reference_features)
            baselines = torch.stack([reference_inputs[i - length + 1:i + 1] for i in reference_positions])
        else:
            baselines = self._inputs(reference_features.iloc[reference_positions])
        inputs = self._inputs(features)
        positions = np.arange(len(features)) if positions is None else np.asarray(positions, dtype=int)
        if not len(positions) or (positions < 0).any() or (positions >= len(features)).any():
            raise ValueError("Attribution positions must identify existing input rows")
        if sample_size is not None and len(positions) > sample_size:
            positions = np.sort(np.random.default_rng(self.random_seed).choice(positions, sample_size, replace=False))
        if self.requires_sequence:
            self._check_timeline(features)
            count = self.sequence_length - 1
            context = self.context_frame if history is None else history
            if count:
                validate_context(context.tail(count), features, self.interval_minutes)
            prior = self.training_context if history is None else self._inputs(history.tail(count) if count else history.iloc[:0])
            windows = ContextSequenceDataset(prior, inputs, self.sequence_length)
            inputs = torch.stack([windows[int(i)][0] for i in positions])
        else:
            inputs = inputs[positions]
        if device is None:
            if self.device.type in {"cuda", "xpu"}:
                device = self.device
            elif torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"
        device = torch.device(device)
        if device.type not in {"cpu", "cuda", "xpu"}:
            raise ValueError("GradientShap device must be cpu, cuda, or xpu")
        dtype = torch.float64 if device.type == "cpu" else torch.float32
        inputs = inputs.to(device=device, dtype=dtype)
        baselines = baselines.to(device=device, dtype=dtype)
        network = copy.deepcopy(self.network).to(device=device, dtype=dtype).eval()
        network.requires_grad_(False)
        batch_size = max(1, 32 // length)
        draw_batch_size = n_samples if device.type == "cpu" else min(n_samples, 128)
        backend = getattr(torch, device.type) if device.type != "cpu" else None
        gpu_devices = (
            [device.index if device.index is not None else backend.current_device()]
            if backend is not None else []
        )
        output_index = target
        target = self.target_transformer

        def output(x):
            values = network(x)
            values = values[:, output_index] if output_index is not None else values.reshape(-1)
            z = values * target.std_ + target.mean_
            return target.scale * torch.sinh(z)

        explainer = GradientShap(output)
        contributions = []
        # Captum samples with NumPy and Torch; make attribution repeatable without
        # changing the random state used by the rest of the pipeline.
        numpy_state = np.random.get_state()
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.random.fork_rng(
                    devices=gpu_devices,
                    device_type=device.type if backend is not None else "cuda",
                ))
                if backend is not None:
                    stack.enter_context(backend.device(device))
                    backend.manual_seed(self.random_seed)
                if device.type == "cuda" and any(isinstance(layer, nn.RNNBase) for layer in network.modules()):
                    # Native CUDA RNNs support backward in eval mode; cuDNN RNNs do not.
                    stack.enter_context(torch.backends.cudnn.flags(enabled=False))
                np.random.seed(self.random_seed)
                torch.random.default_generator.manual_seed(self.random_seed)
                for batch in inputs.split(batch_size):
                    attribution = torch.zeros_like(batch)
                    # Weight the final partial chunk so every requested draw counts equally.
                    for start in range(0, n_samples, draw_batch_size):
                        draws = min(draw_batch_size, n_samples - start)
                        chunk = explainer.attribute(batch, baselines=baselines, n_samples=draws, stdevs=0.0)
                        attribution.add_(chunk.detach(), alpha=draws / n_samples)
                    contributions.append(attribution.cpu().numpy())
        finally:
            np.random.set_state(numpy_state)
        values = np.concatenate(contributions)
        totals = values.sum(axis=1) if self.requires_sequence else values
        index = features.index.take(positions)
        result = pd.DataFrame(totals, index=index, columns=self.feature_columns)
        absolute = np.abs(values).sum(axis=1) if self.requires_sequence else np.abs(values)
        result.attrs["absolute_contributions"] = pd.DataFrame(absolute, index=index, columns=self.feature_columns)
        if self.requires_sequence:
            columns = [f"lag_{lag}_{feature}" for lag in range(self.sequence_length - 1, -1, -1) for feature in self.feature_columns]
            result.attrs["lag_attributions"] = pd.DataFrame(values.reshape(len(values), -1), index=index, columns=columns)
        with torch.no_grad():
            reference_prediction = torch.cat([output(batch).cpu() for batch in baselines.split(batch_size)]).mean().item()
            difference = torch.cat([output(batch).cpu() for batch in inputs.split(batch_size)]).numpy() - reference_prediction
        result.attrs["device"] = str(inputs.device)
        result.attrs["target_index"] = output_index
        result.attrs["target_column"] = (
            self.target_columns[output_index]
            if output_index is not None and self.target_columns is not None else None
        )
        result.attrs["reference_prediction"] = reference_prediction
        result.attrs["baseline_count"] = len(baselines)
        result.attrs["completeness_residual"] = difference - totals.sum(axis=1)
        return result

    def train(self, x_train, y_train):
        """Fit for max_epochs without validation, as used in the final refit."""
        self._train(x_train, y_train)

    def train_with_validation(self, x_train, y_train, x_validation, y_validation):
        """Fit and restore the checkpoint selected by chronological validation."""
        return self._train(
            x_train, y_train, x_validation=x_validation, y_validation=y_validation
        )

    def _train(self, x_train, y_train, *, x_validation=None, y_validation=None):
        """Fit transforms, build samples, then delegate epoch selection to the trainer.

        NaN targets mark rows excluded from the loss. Their features remain in
        sequence windows, so label embargoes do not create artificial time gaps.
        """
        if x_train.empty or len(x_train) != len(y_train):
            raise ValueError(
                "Training inputs and targets must be nonempty and have equal lengths"
            )
        if not x_train.index.equals(y_train.index):
            raise ValueError("Training input and target indices must match")
        if (x_validation is None) != (y_validation is None):
            raise ValueError("Validation inputs and targets must both be provided")
        if self.requires_sequence:
            self._check_timeline(x_train)
        if isinstance(y_train, pd.DataFrame):
            if y_train.shape[1] == 0 or y_train.columns.has_duplicates:
                raise ValueError("Target columns must be nonempty and unique")
            target = self._numeric_frame(y_train)
            self.output_size = target.shape[1]
            self.target_columns = list(target.columns)
            valid = target.notna().all(axis=1).to_numpy(copy=True)
        else:
            target = pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
            self.output_size = 1
            self.target_columns = None
            valid = target.notna().to_numpy(copy=True)
        if not valid.any():
            raise ValueError("y_train contains no finite targets")

        self.feature_columns = list(x_train.columns)
        # Fit transformations only on eligible fitting rows. Unlabeled/embargo
        # rows are transformed using these fitted statistics for context.
        fitting = self._numeric_frame(x_train.loc[valid])
        self.feature_medians = fitting.median().fillna(0.0)
        groups = suggested_scaling_groups(self.feature_columns)
        self.preprocessor = NumericPreprocessor(
            asinh_columns=tuple(groups["asinh_then_standardise"]),
            standardise_columns=tuple(groups["standardise_only"]),
        ).fit(fitting.fillna(self.feature_medians))
        self.target_transformer = TargetTransformer().fit(target.loc[valid])
        inputs = self._inputs(x_train)
        targets = torch.tensor(
            self.target_transformer.transform(target), dtype=torch.float32
        ).reshape(len(target), self.output_size)
        if self.requires_sequence:
            dataset = SequenceDataset(
                inputs, targets, self.sequence_length, target_mask=valid
            )
            count = self.sequence_length - 1
            # Retain both forms: tensors for inference and indexed features for
            # checking whether a later prediction call follows this history.
            self.training_context = (
                inputs[-count:].clone() if count else inputs[:0].clone()
            )
            self.context_frame = x_train.tail(count).copy()
        else:
            dataset = TensorDataset(inputs[valid], targets[valid])

        validation_inputs = validation_actual = None
        if x_validation is not None:
            if x_validation.empty or len(x_validation) != len(y_validation):
                raise ValueError(
                    "Validation inputs and targets must be nonempty and have equal lengths"
                )
            if not x_validation.index.equals(y_validation.index):
                raise ValueError("Validation input and target indices must match")
            if self.requires_sequence:
                self._check_timeline(x_validation)
                if self.sequence_length > 1:
                    validate_context(
                        self.context_frame, x_validation, self.interval_minutes
                    )
            validation_inputs = self._inputs(x_validation)
            if self.target_columns is not None:
                if not isinstance(y_validation, pd.DataFrame) or list(y_validation.columns) != self.target_columns:
                    raise ValueError("Validation target columns must match training target columns in order")
                validation_actual = self._numeric_frame(y_validation).to_numpy(dtype=float)
                if self.output_size == 1:
                    validation_actual = validation_actual[:, 0]
            else:
                validation_actual = pd.to_numeric(y_validation, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(validation_actual)
            if not (finite.all(axis=1) if finite.ndim == 2 else finite).any():
                raise ValueError("y_validation contains no finite targets")

        self._set_random_seed()
        self.network = self._build_network(len(self.feature_columns)).to(self.device)
        # Shuffle complete samples, never timesteps within a window. Each sample
        # is independent; these networks do not carry hidden state between batches.
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = None
        if self.scheduler_step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        criterion = self._build_criterion()
        amp_enabled = bool(self.mixed_precision and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        epoch_number: int = 0

        def train_epoch() -> float:
            """Run one pass over fitting samples and return their mean loss."""
            nonlocal epoch_number
            epoch_number += 1
            self.network.train()
            total = 0.0
            with ExitStack() as stack:
                batches: Iterable[Sequence[torch.Tensor]] = loader
                if self.show_progress:
                    batches = stack.enter_context(
                        tqdm(
                            loader,
                            total=len(loader),
                            desc=f"{self.get_model_name()} epoch {epoch_number}/{self.max_epochs}",
                            leave=False
                        )
                    )
                for xb, yb in batches:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                        prediction = self.network(xb)
                        if prediction.shape != yb.shape:
                            raise ValueError(
                                f"Network output {prediction.shape} must match targets {yb.shape}"
                            )
                        loss = criterion(prediction, yb)
                    scaler.scale(loss).backward()
                    if self.gradient_clip_norm is not None:
                        # AMP scales gradients to protect small values. Undo that
                        # scaling before clipping so the threshold has its usual units.
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.network.parameters(), self.gradient_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    total += loss.item() * len(xb)
            # Weight by batch size above so a short final batch is not overcounted.
            if scheduler is not None:
                scheduler.step()
            return total / len(dataset)

        callback = None
        if validation_inputs is not None:
            def callback():
                # Evaluate on the original target scale for checkpoint selection.
                if self.selection_metric == "mae":
                    return self._validation_mae(validation_inputs, validation_actual)
                return self._validation_score(
                    validation_inputs, validation_actual, self.selection_metric
                )
        result = TorchEarlyStoppingTrainer(
            max_epochs=self.max_epochs, patience=self.patience, min_delta=self.min_delta
        ).fit(network=self.network, train_epoch=train_epoch, validation_score=callback)
        self.training_result = result
        self.training_loss_history = result.training_loss_history
        self.validation_score_history = result.validation_score_history
        self.validation_mae_history = (
            result.validation_score_history if self.selection_metric == "mae" else []
        )
        self.best_epoch, self.stopped_epoch = result.best_epoch, result.stopped_epoch
        return result

    def _predict_transformed(self, inputs, history=None):
        """Run batched inference in transformed target space, preserving row order.

        Inputs are already scaled tensors. ``history`` overrides the saved tail
        for this call only; it does not advance or modify the model's context.
        """
        if self.network is None:
            raise RuntimeError("train() must be called before predict()")
        if not len(inputs):
            return np.empty((0, self.output_size) if self.output_size > 1 else 0, dtype=float)
        dataset = (
            ContextSequenceDataset(
                self.training_context if history is None else history,
                inputs,
                self.sequence_length,
            )
            if self.requires_sequence
            else TensorDataset(inputs)
        )
        loader = DataLoader(
            dataset, batch_size=self.prediction_batch_size, shuffle=False
        )
        self.network.eval()
        batches = []
        with torch.inference_mode():
            for (xb,) in loader:
                batches.append(
                    self._forecast_values(
                        self.network(xb.to(self.device)).cpu().numpy(), len(xb)
                    )
                )
        return np.concatenate(batches)

    def _validation_score(self, inputs, actual, metric):
        """Evaluate in original target units so scores are comparable across models."""
        prediction = self.target_transformer.inverse_transform(
            self._predict_transformed(inputs)
        )
        valid = np.isfinite(actual)
        if valid.ndim == 2:
            valid = valid.all(axis=1)
        if not np.isfinite(prediction[valid]).all():
            raise RuntimeError("Validation produced non-finite predictions")
        return score_forecast(actual[valid].ravel(), prediction[valid].ravel(), metric)

    def _validation_mae(self, inputs, actual):
        return self._validation_score(inputs, actual, "mae")

    def predict(self, x_test, *, history=None):
        """Return target-scale forecasts in row and fitted target-column order.

        For sequence models, history contains observed feature rows immediately
        preceding x_test. Pass it explicitly for successive prediction chunks;
        otherwise the stored training tail is used. No targets are read here,
        and prediction does not update model weights, scalers, or saved history.
        """
        if self.network is None or self.preprocessor is None:
            raise RuntimeError("train() must be called before predict()")
        if x_test.empty:
            return np.empty((0, self.output_size) if self.output_size > 1 else 0, dtype=float)
        history_inputs = None
        if self.requires_sequence:
            self._check_timeline(x_test)
            count = self.sequence_length - 1
            context = self.context_frame if history is None else history
            if count:
                # Validate all supplied history, including future/duplicate rows.
                self._check_timeline(context)
                validate_context(context.tail(count), x_test, self.interval_minutes)
            history_inputs = (
                self.training_context
                if history is None
                else self._inputs(context.tail(count))
            )
        transformed = self._predict_transformed(self._inputs(x_test), history_inputs)
        return self._forecast_values(
            self.target_transformer.inverse_transform(transformed), len(x_test)
        )
