from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pipeline.training import TrainingResult


class Model(ABC):
    model_name: str
    requires_sequence: bool = False

    @abstractmethod
    def __init__(self, **model_params) -> None:
        pass

    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame) -> None:
        pass

    def train_with_validation(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        x_validation: pd.DataFrame,
        y_validation: pd.Series | pd.DataFrame,
    ) -> TrainingResult | None:
        """Train with optional validation support.

        Models without an internal validation procedure retain the ordinary
        training behaviour. Models such as the MLP can override this method
        to implement checkpointing and early stopping.
        """
        self.train(x_train, y_train)
        return None

    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        pass

    def get_model_name(self) -> str:
        return self.model_name
