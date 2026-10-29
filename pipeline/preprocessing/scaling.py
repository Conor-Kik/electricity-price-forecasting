from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class NumericPreprocessor:
    """Small train-only numeric transformer for linear/deep models.

    Tree models should normally skip this class. Heavy-tailed price-like columns can
    receive asinh before standardisation. The transformer stores only training-set
    statistics and can therefore be safely reused on validation/test/live data.
    """

    asinh_columns: tuple[str, ...] = ()
    standardise_columns: tuple[str, ...] = ()
    asinh_scale: float = 100.0
    eps: float = 1e-8
    means_: dict[str, float] = field(default_factory=dict, init=False)
    stds_: dict[str, float] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame) -> "NumericPreprocessor":
        work = self._asinh(df.copy())
        cols = list(dict.fromkeys([*self.asinh_columns, *self.standardise_columns]))
        self.means_ = {}
        self.stds_ = {}
        for col in cols:
            if col not in work.columns:
                continue
            s = pd.to_numeric(work[col], errors="coerce")
            mean = float(s.mean())
            std = float(s.std(ddof=0))
            self.means_[col] = mean
            self.stds_[col] = std if np.isfinite(std) and std > self.eps else 1.0
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("NumericPreprocessor.fit() must be called first")
        out = self._asinh(df.copy())
        for col, mean in self.means_.items():
            if col in out.columns:
                out[col] = (pd.to_numeric(out[col], errors="coerce") - mean) / self.stds_[col]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _asinh(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.asinh_columns:
            if col in df.columns:
                df[col] = np.arcsinh(pd.to_numeric(df[col], errors="coerce") / self.asinh_scale)
        return df


def suggested_scaling_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Conservative defaults for neural/linear models.

    Prices and price deltas are heavy-tailed. Ordinary MW quantities are generally
    standardised only. Bounded cyclic/binary features are left unchanged.
    """
    no_scale: list[str] = []
    asinh: list[str] = []
    standardise: list[str] = []

    for col in feature_columns:
        if col.endswith(("_sin", "_cos")) or col.startswith("is_"):
            no_scale.append(col)
        elif "rrp" in col or "price" in col:
            asinh.append(col)
        else:
            standardise.append(col)

    return {
        "asinh_then_standardise": asinh,
        "standardise_only": standardise,
        "no_scaling": no_scale,
    }


@dataclass
class TargetTransformer:
    """Optional heavy-tail transform for neural-network price targets.

    Tree baselines should normally start with raw $/MWh targets. For deep models,
    this provides the asinh + train-only standardisation option discussed in the
    modelling design, including an exact inverse transform back to $/MWh.
    """

    scale: float = 100.0
    standardise: bool = True
    mean_: float = field(default=0.0, init=False)
    std_: float = field(default=1.0, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, y: pd.Series | np.ndarray) -> "TargetTransformer":
        values = np.asarray(y, dtype=float)
        transformed = np.arcsinh(values / self.scale)
        finite = transformed[np.isfinite(transformed)]
        if finite.size == 0:
            raise ValueError("Target contains no finite values")
        self.mean_ = float(finite.mean()) if self.standardise else 0.0
        std = float(finite.std()) if self.standardise else 1.0
        self.std_ = std if np.isfinite(std) and std > 1e-8 else 1.0
        self.fitted_ = True
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TargetTransformer.fit() must be called first")
        transformed = np.arcsinh(np.asarray(y, dtype=float) / self.scale)
        return (transformed - self.mean_) / self.std_

    def inverse_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TargetTransformer.fit() must be called first")
        transformed = np.asarray(y, dtype=float) * self.std_ + self.mean_
        return self.scale * np.sinh(transformed)

    def fit_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)
