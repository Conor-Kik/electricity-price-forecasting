from __future__ import annotations

import importlib
import importlib.util
import inspect
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pipeline.neural import TorchForecastModel  # noqa: E402
import pipeline.neural as neural  # noqa: E402


class RecordingProgress:
    """Record progress-bar use while yielding the original training batches."""

    instances: ClassVar[list[RecordingProgress]] = []

    def __init__(self, iterable: Iterable[object], **options: object) -> None:
        """Keep the batch iterable and the requested display settings."""
        self.iterable: Iterable[object] = iterable
        self.options: dict[str, object] = options
        self.items_seen: int = 0
        self.closed: bool = False
        self.instances.append(self)

    def __enter__(self) -> RecordingProgress:
        """Support a context-managed progress bar."""
        return self

    def __exit__(self, *_: object) -> None:
        """Record that the bar closed after an epoch."""
        self.closed = True

    def __iter__(self) -> Iterator[object]:
        """Yield each original batch while counting completed batches."""
        for item in self.iterable:
            self.items_seen += 1
            yield item


@pytest.fixture
def progress_bars(monkeypatch: pytest.MonkeyPatch) -> list[RecordingProgress]:
    """Replace tqdm so tests can inspect progress without terminal output."""
    RecordingProgress.instances.clear()
    monkeypatch.setattr(neural, "tqdm", RecordingProgress, raising=False)
    return RecordingProgress.instances


def load_network_type():
    name = "electricity_price_forecasting_model.layers.tcn"
    assert importlib.util.find_spec(name) is not None, "TCN network module is missing"
    return importlib.import_module(name).TCNNetwork


def load_model_type():
    name = "electricity_price_forecasting_model.final_model.tcn"
    assert importlib.util.find_spec(name) is not None, "TCN model module is missing"
    return importlib.import_module(name).TCNModel


def test_tcn_network_preserves_time_and_returns_one_forecast():
    network_type = load_network_type()
    network = network_type(
        input_size=3, channels=8, kernel_size=3, num_blocks=2, dropout=0.0
    )
    inputs = torch.randn(4, 12, 3)

    assert network.temporal_features(inputs).shape == (4, 8, 12)
    assert network(inputs).shape == (4, 1)
    assert network(torch.randn(2, 5, 3)).shape == (2, 1)
    assert network.receptive_field == 13
    dilations = [
        layer.dilation[0]
        for layer in network.modules()
        if isinstance(layer, torch.nn.Conv1d) and layer.kernel_size == (3,)
    ]
    assert dilations == [1, 1, 2, 2]


def test_tcn_temporal_features_are_causal():
    network_type = load_network_type()
    torch.manual_seed(7)
    network = network_type(
        input_size=2, channels=4, kernel_size=3, num_blocks=2, dropout=0.0
    )
    network.eval()
    inputs = torch.randn(2, 12, 2)
    changed = inputs.clone()
    changed[:, 7:, :] += 1000

    with torch.inference_mode():
        original_prefix = network.temporal_features(inputs)[:, :, :7]
        changed_prefix = network.temporal_features(changed)[:, :, :7]
        truncated_prefix = network.temporal_features(inputs[:, :7, :])

    torch.testing.assert_close(original_prefix, changed_prefix)
    torch.testing.assert_close(original_prefix, truncated_prefix)


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_tcn_offset_batches_preserve_predictions_and_input_gradients(device):
    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip("XPU is unavailable")
    network = load_network_type()(input_size=7, channels=16).to(device).eval()
    storage = torch.randn(4, 12, 7, device=device)
    # Offset by one float: contiguous, but not 64-byte aligned.
    flat = torch.randn(1 + storage.numel(), device=device)
    inputs = flat[1:].view_as(storage).detach().requires_grad_(True)
    reference = inputs.detach().clone().requires_grad_(True)
    assert inputs.is_contiguous() and inputs.data_ptr() % 64 != 0

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Tensor is not 64-byte aligned.*")
        actual = network(inputs)
        expected = network(reference)
        actual_grad = torch.autograd.grad(actual.sum(), inputs)[0]
        expected_grad = torch.autograd.grad(expected.sum(), reference)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_tcn_model_exposes_the_shared_sequence_contract():
    model_type = load_model_type()
    network_type = load_network_type()
    model = model_type(
        sequence_length=12,
        channels=16,
        kernel_size=3,
        num_blocks=2,
        dropout=0.1,
        learning_rate=3e-4,
        batch_size=8,
        max_epochs=2,
        device="cpu",
    )

    assert isinstance(model, TorchForecastModel)
    assert model.get_model_name() == "TCN"
    assert model.requires_sequence is True
    assert model.sequence_length == 12
    assert model.channels == 16
    assert model.kernel_size == 3
    assert model.num_blocks == 2
    assert model.dropout == pytest.approx(0.1)
    assert model.learning_rate == pytest.approx(3e-4)
    assert isinstance(model._build_network(input_size=2), network_type)


@pytest.mark.parametrize(
    "settings",
    [
        {"sequence_length": 0},
        {"channels": 0},
        {"kernel_size": 1},
        {"num_blocks": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_tcn_model_rejects_invalid_architecture_settings(settings):
    model_type = load_model_type()
    with pytest.raises(ValueError):
        model_type(**settings)


def test_tcn_trains_and_predicts_with_explicit_history():
    model_type = load_model_type()
    network_type = load_network_type()
    x_train = pd.DataFrame(
        {
            "price_lag": [10.0, 12.0, np.nan, 16.0, 18.0, 20.0, 22.0, 24.0],
            "demand": [100.0, 105.0, 110.0, np.nan, 120.0, 125.0, 130.0, 135.0],
        }
    )
    y_train = pd.Series([11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0])
    current = pd.DataFrame(
        {"price_lag": [26.0, np.nan, 30.0], "demand": [140.0, 145.0, 150.0]}
    )
    model = model_type(
        sequence_length=3,
        channels=4,
        kernel_size=3,
        num_blocks=1,
        batch_size=2,
        max_epochs=1,
        random_seed=7,
        device="cpu",
    )

    model.train(x_train, y_train)
    all_at_once = model.predict(current, history=x_train)
    first = model.predict(current.iloc[:1], history=x_train)
    later_history = pd.concat([x_train, current.iloc[:1]])
    rest = model.predict(current.iloc[1:], history=later_history)

    assert isinstance(model.network, network_type)
    assert all_at_once.shape == (3,)
    assert np.isfinite(all_at_once).all()
    assert len(model.training_context) == 2
    assert len(model.training_loss_history) == 1
    np.testing.assert_allclose(all_at_once, np.concatenate([first, rest]))


def progress_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return enough rows for three batches of length-three windows."""
    features: pd.DataFrame = pd.DataFrame(
        {"price": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0, 17.0, 16.0]}
    )
    targets: pd.Series = pd.Series(
        [11.0, 12.0, 14.0, 13.0, 15.0, 16.0, 18.0, 17.0]
    )
    return features, targets


def test_tcn_progress_reports_batches_and_epochs(
    progress_bars: list[RecordingProgress],
) -> None:
    """Show one closed batch bar for each completed training epoch."""
    model_type = load_model_type()
    model = model_type(
        sequence_length=3,
        channels=4,
        num_blocks=1,
        batch_size=2,
        max_epochs=2,
        device="cpu",
    )

    model.train(*progress_training_data())

    assert len(progress_bars) == 2
    assert [bar.options["desc"] for bar in progress_bars] == [
        "TCN epoch 1/2",
        "TCN epoch 2/2",
    ]
    assert all(bar.options["total"] == 3 for bar in progress_bars)
    assert all(bar.items_seen == 3 and bar.closed for bar in progress_bars)


def test_tcn_progress_can_be_disabled(
    progress_bars: list[RecordingProgress],
) -> None:
    """Keep quiet runs free of progress-bar construction."""
    model_type = load_model_type()
    assert "show_progress" in inspect.signature(model_type).parameters
    model = model_type(
        sequence_length=3,
        channels=4,
        num_blocks=1,
        batch_size=2,
        max_epochs=1,
        show_progress=False,
        device="cpu",
    )

    model.train(*progress_training_data())

    assert progress_bars == []


@pytest.mark.parametrize("show_progress", [True, False])
def test_other_neural_models_use_shared_progress_setting(
    progress_bars: list[RecordingProgress],
    show_progress: bool,
) -> None:
    """The shared trainer respects the progress setting for other models too."""
    from electricity_price_forecasting_model.final_model.gru import GRUModel

    model: GRUModel = GRUModel(
        sequence_length=3,
        hidden_size=4,
        batch_size=2,
        max_epochs=1,
        device="cpu",
    )
    model.show_progress = show_progress

    model.train(*progress_training_data())

    assert len(progress_bars) == int(show_progress)
    if show_progress:
        bar = progress_bars[0]
        assert bar.options["desc"] == "GRU epoch 1/1"
        assert bar.items_seen == 3 and bar.closed


def test_tcn_progress_does_not_change_training_result(
    progress_bars: list[RecordingProgress],
) -> None:
    """Preserve seeded losses and forecasts when display is enabled."""
    model_type = load_model_type()
    assert "show_progress" in inspect.signature(model_type).parameters
    settings: dict[str, object] = {
        "sequence_length": 3,
        "channels": 4,
        "num_blocks": 1,
        "batch_size": 2,
        "max_epochs": 1,
        "random_seed": 7,
        "device": "cpu",
    }
    features, targets = progress_training_data()
    visible = model_type(**settings, show_progress=True)
    quiet = model_type(**settings, show_progress=False)

    visible.train(features, targets)
    quiet.train(features, targets)

    assert len(progress_bars) == 1
    np.testing.assert_allclose(
        visible.training_loss_history, quiet.training_loss_history
    )
    np.testing.assert_allclose(
        visible.predict(features.iloc[-2:]), quiet.predict(features.iloc[-2:])
    )


def test_tqdm_is_a_direct_dependency() -> None:
    """Make progress display available after a fresh requirements install."""
    requirements: Path = Path(__file__).resolve().parents[1] / "requirements.txt"
    lines: list[str] = requirements.read_text().splitlines()

    assert any(line.strip().startswith("tqdm") for line in lines)
