"""Кастомные трансформеры для варианта 17."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgeSquaredTransformer(BaseEstimator, TransformerMixin):
    """Добавляет признак age^2 (вариант 17: «возраст в квадрате»)."""

    def __init__(self, age_col: str = "age", out_col: str = "age_squared"):
        self.age_col = age_col
        self.out_col = out_col

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame) and self.age_col not in X.columns:
            raise ValueError(f"Колонка '{self.age_col}' отсутствует в данных")
        return self

    def transform(self, X):
        X_out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.age_col not in X_out.columns:
            raise ValueError(f"Колонка '{self.age_col}' отсутствует в данных")
        X_out[self.out_col] = X_out[self.age_col].astype(float) ** 2
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return [self.out_col]
        names = list(input_features)
        if self.out_col not in names:
            names.append(self.out_col)
        return names
