"""Лабораторная работа 2: импутация, выбросы, отбор признаков, финальный пайплайн."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.transformers import (  # noqa: E402
    AgeSquaredTransformer,
    IsolationForestOutlierRemover,
    NanSafeStandardScaler,
)

RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
MODEL_PATH = ROOT / "models" / "heart_disease_lab2_pipeline.joblib"
METRICS_PATH = REPORTS / "lab2_metrics.json"

NUMERIC_BASE = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"
RANDOM_STATE = 42


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Нет {RAW_PATH}. Запустите: python -m src.data.make_dataset"
        )
    df = pd.read_csv(RAW_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_imputer_pipeline(imputer_name: str) -> Pipeline:
    """Числовые: scale → imputer; категориальные: mode → OHE; затем LR."""
    if imputer_name == "simple":
        num_imputer = SimpleImputer(strategy="median")
    elif imputer_name == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif imputer_name == "iterative":
        num_imputer = IterativeImputer(random_state=RANDOM_STATE, max_iter=20)
    else:
        raise ValueError(imputer_name)

    # для KNN/Iterative — масштабирование до импутации (NaN-safe)
    if imputer_name == "simple":
        numeric = Pipeline(
            steps=[
                ("imputer", num_imputer),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric = Pipeline(
            steps=[
                ("scaler", NanSafeStandardScaler()),
                ("imputer", num_imputer),
            ]
        )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_BASE),
            ("cat", categorical, CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("age_squared", AgeSquaredTransformer()),
            ("pre", pre),
            (
                "model",
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            ),
        ]
    )


def compare_imputers(X_train, y_train) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name in ["simple", "knn", "iterative"]:
        pipe = build_imputer_pipeline(name)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        rows.append(
            {
                "imputer": name,
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
            }
        )
        print(f"  {name}: ROC-AUC={scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows)


def plot_imputation_hist(X_train: pd.DataFrame) -> Path:
    col = "chol"
    before = X_train[col].dropna()
    # сравнение заполнений
    simple = SimpleImputer(strategy="median").fit_transform(
        X_train[[col]]
    ).ravel()
    knn = Pipeline(
        [("sc", NanSafeStandardScaler()), ("imp", KNNImputer(n_neighbors=5))]
    ).fit_transform(X_train[NUMERIC_BASE])
    knn_col = knn[:, NUMERIC_BASE.index(col)]
    # вернуть chol в исходный масштаб для knn — берём только визуально imputed gaps
    # проще: гистограмма observed + imputed-only точек для simple
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    axes[0].hist(before, bins=20, color="#4C78A8", alpha=0.85, edgecolor="white")
    axes[0].set_title(f"{col}: до (без NaN)")
    axes[0].set_xlabel(col)

    axes[1].hist(simple, bins=20, color="#F58518", alpha=0.85, edgecolor="white")
    axes[1].set_title(f"{col}: SimpleImputer (median)")
    axes[1].set_xlabel(col)

    it = Pipeline(
        [
            ("sc", NanSafeStandardScaler()),
            ("imp", IterativeImputer(random_state=RANDOM_STATE, max_iter=20)),
        ]
    ).fit_transform(X_train[NUMERIC_BASE])
    axes[2].hist(
        knn_col, bins=20, color="#54A24B", alpha=0.55, edgecolor="white", label="KNN"
    )
    axes[2].hist(
        it[:, NUMERIC_BASE.index(col)],
        bins=20,
        color="#B279A2",
        alpha=0.55,
        edgecolor="white",
        label="Iterative",
    )
    axes[2].set_title(f"{col}: KNN / Iterative (scaled)")
    axes[2].set_xlabel(f"{col} (z-score)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "lab2_imputation_chol.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def detect_and_plot_outliers(X_train: pd.DataFrame) -> dict:
    X_num = SimpleImputer(strategy="median").fit_transform(X_train[NUMERIC_BASE])
    X_scaled = StandardScaler().fit_transform(X_num)
    iso = IsolationForest(
        contamination=0.05, random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1
    )
    labels = iso.fit_predict(X_scaled)
    n_out = int((labels == -1).sum())

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    inliers = labels == 1
    ax.scatter(
        coords[inliers, 0],
        coords[inliers, 1],
        c="#4C78A8",
        s=28,
        alpha=0.75,
        label=f"inliers ({inliers.sum()})",
    )
    ax.scatter(
        coords[~inliers, 0],
        coords[~inliers, 1],
        c="#E45756",
        s=48,
        alpha=0.9,
        marker="X",
        label=f"outliers ({n_out})",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("IsolationForest → PCA(2D)")
    ax.legend()
    ax.grid(alpha=0.3)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "lab2_outliers_pca.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return {"n_outliers": n_out, "figure": str(path)}


def build_feature_matrix(X: pd.DataFrame, y: pd.Series | None = None, fit_pre=None):
    """Препроцессинг + age² + PolynomialFeatures(degree=2) на числовых."""
    numeric_with_sq = NUMERIC_BASE + ["age_squared"]
    age = AgeSquaredTransformer()
    X2 = age.fit_transform(X)

    num_pipe = Pipeline(
        [
            ("scaler", NanSafeStandardScaler()),
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, numeric_with_sq),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )
    if fit_pre is None:
        Xt = pre.fit_transform(X2)
        return Xt, pre, age
    Xt = pre.transform(age.transform(X))
    return Xt, fit_pre, age


def compare_feature_selection(X_train, y_train) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    Xt, pre, _ = build_feature_matrix(X_train, y_train)
    n_features = Xt.shape[1]
    k = min(30, n_features)

    clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)

    # baseline: все признаки после poly
    base_scores = cross_val_score(clf, Xt, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    # Mutual Information
    mi_pipe = Pipeline(
        [
            ("sel", SelectKBest(mutual_info_classif, k=k)),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )
    mi_scores = cross_val_score(mi_pipe, Xt, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    # Lasso (L1)
    lasso_pipe = Pipeline(
        [
            (
                "sel",
                SelectFromModel(
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        l1_ratio=1.0,
                        C=0.5,
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )
    lasso_scores = cross_val_score(
        lasso_pipe, Xt, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    rows = [
        {
            "method": "all_poly_features",
            "n_features": int(n_features),
            "cv_roc_auc_mean": float(base_scores.mean()),
            "cv_roc_auc_std": float(base_scores.std()),
        },
        {
            "method": "mutual_info_SelectKBest",
            "n_features": int(k),
            "cv_roc_auc_mean": float(mi_scores.mean()),
            "cv_roc_auc_std": float(mi_scores.std()),
        },
        {
            "method": "lasso_SelectFromModel",
            "n_features": "adaptive",
            "cv_roc_auc_mean": float(lasso_scores.mean()),
            "cv_roc_auc_std": float(lasso_scores.std()),
        },
    ]
    for r in rows:
        print(
            f"  {r['method']}: ROC-AUC={r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}"
        )
    return pd.DataFrame(rows)


def build_final_pipeline() -> ImbPipeline:
    numeric_with_sq = NUMERIC_BASE + ["age_squared"]
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )
    num_pipe = Pipeline(
        [
            ("scaler", NanSafeStandardScaler()),
            ("imputer", IterativeImputer(random_state=RANDOM_STATE, max_iter=20)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num_pipe, numeric_with_sq),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )
    return ImbPipeline(
        steps=[
            ("age_squared", AgeSquaredTransformer()),
            (
                "outlier_remover",
                IsolationForestOutlierRemover(
                    contamination=0.05,
                    random_state=RANDOM_STATE,
                    numeric_cols=NUMERIC_BASE,
                ),
            ),
            ("pre", pre),
            (
                "select",
                SelectFromModel(
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        l1_ratio=1.0,
                        C=0.5,
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
            (
                "model",
                LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train_final(X_train, y_train, X_test, y_test) -> dict:
    pipe = build_final_pipeline()
    param_grid = {
        "model__C": [0.1, 0.5, 1.0, 2.0],
        "outlier_remover__contamination": [0.03, 0.05, 0.08],
    }
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = {
        "best_params": search.best_params_,
        "cv_best_roc_auc": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }
    print("best_params:", search.best_params_)
    print(f"CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test ROC-AUC:  {metrics['test_roc_auc']:.4f}")
    print(classification_report(y_test, y_pred))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix (test)")
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("ROC curve (test)")
    plt.tight_layout()
    fig_path = FIG_DIR / "lab2_final_metrics.png"
    plt.savefig(fig_path, dpi=140)
    plt.close()
    metrics["figures"] = {"final": str(fig_path)}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    metrics["model_path"] = str(MODEL_PATH)
    return metrics


def main() -> None:
    sns.set_theme(style="whitegrid")
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_xy()
    print(f"Данные: {X.shape}, пропусков: {int(X.isna().sum().sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"split: train={len(X_train)}, test={len(X_test)}")

    print("\n=== 1. Сравнение импутеров ===")
    imp_df = compare_imputers(X_train, y_train)
    imp_fig = plot_imputation_hist(X_train)
    print(f"figure: {imp_fig}")

    print("\n=== 2. IsolationForest + PCA ===")
    out_info = detect_and_plot_outliers(X_train)
    print(f"outliers: {out_info['n_outliers']}, figure: {out_info['figure']}")

    print("\n=== 3. Отбор признаков ===")
    sel_df = compare_feature_selection(X_train, y_train)

    print("\n=== 4. Финальный пайплайн ===")
    final_metrics = train_final(X_train, y_train, X_test, y_test)

    payload = {
        "variant": 17,
        "lab": 2,
        "missingness": "MAR (~12% in chol, trestbps, thalach; depends on age)",
        "outliers_injected": True,
        "imputation_comparison": imp_df.to_dict(orient="records"),
        "outlier_detection": out_info,
        "feature_selection_comparison": sel_df.to_dict(orient="records"),
        "final": final_metrics,
        "figures": {
            "imputation": str(imp_fig),
            "outliers_pca": out_info["figure"],
            "final": final_metrics["figures"]["final"],
        },
    }
    # classification_report содержит numpy types — уже float/dict ok
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nМетрики: {METRICS_PATH}")
    print(f"Модель: {MODEL_PATH}")


if __name__ == "__main__":
    main()
