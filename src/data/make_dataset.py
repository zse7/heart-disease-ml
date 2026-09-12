"""Загрузка Heart Disease (UCI) и подготовка data/raw."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Публичное зеркало Cleveland Heart Disease (UCI)
DATA_URL = (
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
)

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "age_squared"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"


def download_heart_disease(url: str = DATA_URL) -> pd.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    # убрать BOM, если есть
    text = response.content.decode("utf-8-sig")
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    return df


def introduce_missingness(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    fraction: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Искусственно вносит пропуски в ~fraction значений указанных числовых колонок."""
    columns = columns or ["chol", "trestbps"]
    rng = np.random.default_rng(random_state)
    out = df.copy()
    n = len(out)
    n_missing = max(1, int(round(n * fraction)))
    for col in columns:
        idx = rng.choice(n, size=n_missing, replace=False)
        out.loc[out.index[idx], col] = np.nan
    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = download_heart_disease()
    print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Пропусков до: {int(df.isna().sum().sum())}")

    if df.isna().sum().sum() == 0:
        df = introduce_missingness(df, columns=["chol", "trestbps"], fraction=0.10)
        print(
            "Пропусков нет — искусственно внесено ~10% NaN в chol и trestbps. "
            f"Пропусков после: {int(df.isna().sum().sum())}"
        )
    else:
        print("В датасете уже есть пропуски, искусственные не добавлялись.")

    out_path = RAW_DIR / "heart_disease.csv"
    df.to_csv(out_path, index=False)
    print(f"Сохранено: {out_path}")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
