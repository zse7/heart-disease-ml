"""Загрузка Heart Disease (UCI) и подготовка data/raw для лаб 1–2."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

DATA_URL = (
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
)

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "age_squared"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"


def download_heart_disease(url: str = DATA_URL) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.content.decode("utf-8-sig")
    return pd.read_csv(StringIO(text))


def introduce_missingness_mcar(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """MCAR: пропуски равновероятно в указанных колонках."""
    columns = columns or ["chol", "trestbps"]
    rng = np.random.default_rng(random_state)
    out = df.copy()
    n = len(out)
    n_missing = max(1, int(round(n * fraction)))
    for col in columns:
        idx = rng.choice(n, size=n_missing, replace=False)
        out.loc[out.index[idx], col] = np.nan
    return out


def introduce_missingness_mar(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.12,
    random_state: int = 42,
) -> pd.DataFrame:
    """MAR: вероятность пропуска зависит от наблюдаемого age.

    Обоснование: у более возрастных пациентов лабораторные показатели
    (chol, trestbps, thalach) чаще оказываются незаполненными в медкартах —
    пропуск зависит от observed age, а не от самого скрытого значения.
    """
    columns = columns or ["chol", "trestbps", "thalach"]
    rng = np.random.default_rng(random_state)
    out = df.copy()
    age = out["age"].astype(float).to_numpy()
    # веса: старше → выше шанс пропуска
    weights = (age - age.min() + 1.0) ** 1.5
    weights = weights / weights.sum()
    n = len(out)
    n_missing = max(1, int(round(n * fraction)))
    for i, col in enumerate(columns):
        # чуть разный seed на колонку, но воспроизводимо
        local_rng = np.random.default_rng(random_state + i + 1)
        idx = local_rng.choice(n, size=n_missing, replace=False, p=weights)
        out.loc[out.index[idx], col] = np.nan
    return out


def introduce_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_per_col: int = 5,
    factor_range: tuple[float, float] = (5.0, 10.0),
    random_state: int = 42,
) -> pd.DataFrame:
    """Добавляет несколько явных выбросов: значение × 5–10."""
    columns = columns or ["chol", "trestbps", "thalach"]
    rng = np.random.default_rng(random_state)
    out = df.copy()
    for i, col in enumerate(columns):
        local_rng = np.random.default_rng(random_state + 100 + i)
        candidates = out.index[out[col].notna()].to_numpy()
        if len(candidates) == 0:
            continue
        take = min(n_per_col, len(candidates))
        idx = local_rng.choice(candidates, size=take, replace=False)
        factors = local_rng.uniform(factor_range[0], factor_range[1], size=take)
        out.loc[idx, col] = out.loc[idx, col].astype(float).to_numpy() * factors
    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = download_heart_disease()
    print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Пропусков до: {int(df.isna().sum().sum())}")

    # Lab 2: MAR 12% в 3 числовых колонках + искусственные выбросы
    df = introduce_missingness_mar(
        df, columns=["chol", "trestbps", "thalach"], fraction=0.12
    )
    print(
        "MAR: ~12% NaN в chol, trestbps, thalach (зависит от age). "
        f"Пропусков после: {int(df.isna().sum().sum())}"
    )

    df = introduce_outliers(
        df, columns=["chol", "trestbps", "thalach"], n_per_col=5
    )
    print("Добавлены искусственные выбросы (×5–10) в chol / trestbps / thalach.")

    out_path = RAW_DIR / "heart_disease.csv"
    df.to_csv(out_path, index=False)
    print(f"Сохранено: {out_path}")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nЭкстремумы после выбросов:")
    print(df[["chol", "trestbps", "thalach"]].agg(["min", "max"]))


if __name__ == "__main__":
    main()
