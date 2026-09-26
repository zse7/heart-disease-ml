"""Кастомные трансформеры для лабораторных работ."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


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


class MissingCountTransformer(BaseEstimator, TransformerMixin):
    """Добавляет признак «число пропусков в строке»."""

    def __init__(self, out_col: str = "missing_count"):
        self.out_col = out_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_out[self.out_col] = X_out.isna().sum(axis=1).astype(float)
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return [self.out_col]
        names = list(input_features)
        if self.out_col not in names:
            names.append(self.out_col)
        return names


class NanSafeStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler, устойчивый к NaN (статистики по nanmean/nanstd).

    Нужен, чтобы масштабировать признаки *до* KNN/Iterative импутации.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_


class IsolationForestOutlierRemover(BaseEstimator):
    """Удаляет выбросы из обучающей выборки (только fit_resample).

    Для imblearn.pipeline.Pipeline: на fit отбрасывает аномалии
    IsolationForest, на predict шаг сэмплера не применяется —
    тестовые строки не удаляются.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        n_estimators: int = 200,
        numeric_cols: list[str] | None = None,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.numeric_cols = numeric_cols

    def fit_resample(self, X, y):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_arr = np.asarray(y)

        cols = self.numeric_cols
        if cols is None:
            cols = list(X_df.select_dtypes(include=[np.number]).columns)

        X_num = SimpleImputer(strategy="median").fit_transform(X_df[cols])
        detector = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=-1,
        )
        labels = detector.fit_predict(X_num)  # 1 = inlier, -1 = outlier
        mask = labels == 1
        self.n_removed_ = int((~mask).sum())
        self.n_kept_ = int(mask.sum())
        self.detector_ = detector
        return X_df.loc[mask].reset_index(drop=True), y_arr[mask]
