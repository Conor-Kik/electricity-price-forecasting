# ruff: noqa: E402
# Check PyTorch availability before importing dependent modules.
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.preprocessing import make_sequences

torch = pytest.importorskip("torch")

from pipeline.sequences import ContextSequenceDataset, SequenceDataset


def test_training_dataset_keeps_invalid_target_rows_as_later_context():
    inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    targets = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
    dataset = SequenceDataset(
        inputs,
        targets,
        sequence_length=3,
        target_mask=np.array([True, True, False, True, True]),
    )

    assert len(dataset) == 2
    first_window, first_target = dataset[0]
    assert torch.equal(first_window, inputs[1:4])
    assert torch.equal(first_target, targets[3])


def test_context_dataset_returns_one_window_for_every_new_row():
    history = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    current = torch.arange(10, 16, dtype=torch.float32).reshape(3, 2)
    dataset = ContextSequenceDataset(history, current, sequence_length=3)

    assert len(dataset) == len(current)
    assert torch.equal(dataset[0][0], torch.cat((history[-2:], current[:1])))
    assert torch.equal(dataset[2][0], current)


def test_eager_sequences_use_history_without_emitting_history_targets():
    history = pd.DataFrame(
        {
            "region": ["NSW1", "NSW1", "NSW1"],
            "settlement_date": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "feature": [1.0, 2.0, 3.0],
        }
    )
    current = pd.DataFrame(
        {
            "region": ["NSW1", "NSW1"],
            "settlement_date": pd.date_range(
                "2024-01-01 00:15", periods=2, freq="5min"
            ),
            "feature": [4.0, 5.0],
            "target": [40.0, 50.0],
        }
    )

    result = make_sequences(
        current,
        ["feature"],
        target_columns=["target"],
        sequence_length=3,
        history=history,
    )

    np.testing.assert_array_equal(
        result.X[:, :, 0],
        np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(result.y[:, 0], [40.0, 50.0])
    np.testing.assert_array_equal(result.timestamps, current["settlement_date"])


def test_eager_sequences_never_cross_region_boundaries():
    current = pd.DataFrame(
        {
            "region": ["NSW1", "NSW1", "QLD1", "QLD1"],
            "settlement_date": list(pd.date_range("2024-01-01", periods=2, freq="5min"))
            * 2,
            "feature": [1.0, 2.0, 10.0, 20.0],
            "target": [1.0, 2.0, 10.0, 20.0],
        }
    )

    result = make_sequences(
        current,
        ["feature"],
        target_columns=["target"],
        sequence_length=2,
    )

    np.testing.assert_array_equal(
        result.X[:, :, 0],
        np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(result.regions, ["NSW1", "QLD1"])
