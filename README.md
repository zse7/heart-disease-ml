# Heart Disease ML — вариант 17

Воспроизводимый ML-проект по датасету **Heart Disease (UCI / Cleveland)**: Git + DVC, пайплайны `sklearn` / `imblearn`.

| Лаба | Тема | Ноутбук / скрипт |
|------|------|------------------|
| 1 | Pipeline + `age²` | `notebooks/lab1_heart_disease.ipynb`, `src/models/train.py` |
| 2 | Импутация, выбросы, отбор признаков | `notebooks/lab2_feature_engineering.ipynb`, `src/models/train_lab2.py` |

---

## Вариант

| Поле | Значение |
|------|----------|
| № | 17 |
| Датасет | Heart Disease (UCI) |
| Задача | Классификация |
| Целевая переменная | `target` |
| Лаба 1 — трансформер | `age_squared = age²` |
| Лаба 2 — методы | Simple / KNN / Iterative · IsolationForest · MI + Lasso |

## Структура проекта

```text
heart-disease-ml/
├── data/raw/
│   ├── heart_disease.csv          # под DVC
│   └── heart_disease.csv.dvc
├── models/                        # joblib (gitignore)
├── notebooks/
│   ├── lab1_heart_disease.ipynb
│   └── lab2_feature_engineering.ipynb
├── reports/
│   ├── metrics.json               # лаба 1
│   ├── lab2_metrics.json
│   └── figures/                   # графики лабы 2
├── src/
│   ├── data/make_dataset.py
│   ├── features/transformers.py
│   └── models/train.py, train_lab2.py
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
source .venv_heart/bin/activate   # или: python -m venv .venv && pip install -r requirements.txt
pip install imbalanced-learn      # если ещё нет в окружении

python -m src.data.make_dataset   # MAR-пропуски + выбросы (лаба 2)
dvc add data/raw/heart_disease.csv

python -m src.models.train        # лаба 1
python -m src.models.train_lab2   # лаба 2

jupyter lab notebooks/lab2_feature_engineering.ipynb
```

---

# Лабораторная работа 1

Кратко: `AgeSquaredTransformer` + `ColumnTransformer` + `LogisticRegression` / `GridSearchCV`.

| Метрика | Значение |
|---------|----------|
| Лучший `C` | `0.1` |
| CV ROC-AUC | **0.919** |
| Test Accuracy | **0.869** |
| Test ROC-AUC | **0.904** |

Подробности — в истории коммитов / `reports/metrics.json` (метрики лабы 1 на более ранней версии данных с MCAR 10%).

---

# Лабораторная работа 2

## Что сделано по шагам

1. **Проект / DVC** — структура из модуля 1, данные под DVC.
2. **Данные** — Heart Disease в `data/raw`.
   - **MAR ~12%** NaN в `chol`, `trestbps`, `thalach`: вероятность пропуска растёт с `age`  
     *(у возрастных пациентов лабораторные поля чаще незаполнены — пропуск зависит от наблюдаемого возраста)*.
   - **Выбросы:** часть значений ×5–10 в тех же колонках.
3. **Импутация** — сравнение Simple (медиана), KNN, Iterative внутри Pipeline + 5-fold CV ROC-AUC; гистограммы `chol` до/после.
4. **Выбросы** — IsolationForest, визуализация PCA(2D); кастомный `IsolationForestOutlierRemover` (`fit_resample`) в `imblearn.pipeline.Pipeline` — удаляет только из train.
5. **Признаки** — `PolynomialFeatures(degree=2)` (+ `age_squared`); отбор **Mutual Information** и **Lasso/L1** (`SelectFromModel`).
6. **Финал** — полный пайплайн + `GridSearchCV`; оценка на отложенном test.
7. **Артефакты** — `models/heart_disease_lab2_pipeline.joblib`, `reports/lab2_metrics.json`, `reports/figures/`.

## Сравнение импутеров (CV ROC-AUC, train)

| Импутер | Mean ROC-AUC | Std |
|---------|--------------|-----|
| Simple (median) | 0.879 | 0.039 |
| **KNN** | **0.883** | 0.035 |
| Iterative | 0.880 | 0.038 |

Перед KNN/Iterative числовые признаки масштабируются через `NanSafeStandardScaler` (scale до импутации при наличии NaN).

## Отбор признаков (после PolynomialFeatures)

| Метод | CV ROC-AUC |
|-------|------------|
| Все poly-признаки | 0.871 |
| Mutual Information (`SelectKBest`, k=30) | 0.870 |
| **Lasso / L1 (`SelectFromModel`)** | **0.875** |

## Финальная модель (test, n=61)

| Метрика | Значение |
|---------|----------|
| Лучшие параметры | `C=2.0`, `contamination=0.03` |
| CV ROC-AUC | **0.873** |
| Test Accuracy | **0.852** |
| Test ROC-AUC | **0.858** |

Графики: `reports/figures/lab2_imputation_chol.png`, `lab2_outliers_pca.png`, `lab2_final_metrics.png`.

## Выводы (лаба 2)

- **KNN-импутация** дала лучший CV ROC-AUC среди трёх методов (небольшой, но стабильный отрыв).
- IsolationForest находит искусственные и естественные аномалии; remover в imblearn-пайплайне корректно режет **только train**.
- Полиномиальные признаки без отбора на 242 train-строках не помогают; **L1-отбор** слегка улучшает CV и уменьшает шум.
- Итоговый test ROC-AUC ~0.86 ниже лабы 1 — ожидаемо: данные «сложнее» (MAR + выбросы + более тяжёлый конвейер).

## Воспроизведение лабы 2

```bash
python -m src.data.make_dataset
dvc add data/raw/heart_disease.csv
python -m src.models.train_lab2
cat reports/lab2_metrics.json
```
