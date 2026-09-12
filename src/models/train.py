"""Обучение пайплайна для Heart Disease (вариант 17)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.transformers import AgeSquaredTransformer  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "age_squared"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ]
    )
    clf = LogisticRegression(max_iter=2000, random_state=42)
    return Pipeline(
        steps=[
            ("age_squared", AgeSquaredTransformer(age_col="age", out_col="age_squared")),
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Нет файла {RAW_PATH}. Сначала запустите: python -m src.data.make_dataset"
        )

    df = pd.read_csv(RAW_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    param_grid = {
        "model__C": [0.1, 1.0, 10.0],
    }
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "best_params": search.best_params_,
        "cv_best_roc_auc": float(search.best_score_),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Лучшие параметры:", search.best_params_)
    print(f"CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Модель: {MODEL_PATH}")
    print(f"Метрики: {METRICS_PATH}")


if __name__ == "__main__":
    main()
