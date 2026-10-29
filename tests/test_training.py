import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from pipeline.training import TorchEarlyStoppingTrainer  # noqa: E402


def test_trainer_stops_and_restores_best_network_state():
    network = nn.Linear(1, 1, bias=False)
    scores = iter([5.0, 4.0, 4.0, 4.0, 3.0])

    def train_epoch():
        with torch.no_grad():
            network.weight.add_(1.0)
        return float(network.weight.item())

    trainer = TorchEarlyStoppingTrainer(
        max_epochs=10,
        patience=2,
        min_delta=0.001,
    )
    initial_weight = float(network.weight.item())

    result = trainer.fit(
        network=network,
        train_epoch=train_epoch,
        validation_score=lambda: next(scores),
    )

    assert result.best_epoch == 2
    assert result.stopped_epoch == 4
    assert result.best_validation_score == 4.0
    assert result.validation_score_history == [5.0, 4.0, 4.0, 4.0]
    assert network.weight.item() == pytest.approx(initial_weight + 2.0)


def test_trainer_without_validation_runs_every_epoch():
    network = nn.Linear(1, 1)
    trainer = TorchEarlyStoppingTrainer(max_epochs=3, patience=1)

    result = trainer.fit(
        network=network,
        train_epoch=lambda: 1.0,
    )

    assert result.best_epoch == 3
    assert result.stopped_epoch == 3
    assert result.best_validation_score is None
    assert result.training_loss_history == [1.0, 1.0, 1.0]
    assert result.validation_score_history == []
