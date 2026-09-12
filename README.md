# Heart Disease ML — Лабораторная работа 1 (вариант 17)

Воспроизводимый ML-проект по датасету **Heart Disease (UCI / Cleveland)**: Git + DVC, пайплайн `sklearn` с кастомным трансформером «возраст в квадрате».

## Вариант

| Поле | Значение |
|------|----------|
| № | 17 |
| Датасет | Heart Disease (UCI) |
| Задача | Классификация |
| Целевая переменная | `target` (наличие болезни) |
| Кастомный трансформер | Добавляет признак `age_squared = age²` |

## Структура проекта

```text
heart-disease-ml/
├── data/
│   ├── raw/
│   │   ├── heart_disease.csv      # под DVC (не в git)
│   │   └── heart_disease.csv.dvc
│   └── processed/
├── models/                        # joblib-модели (gitignore)
├── notebooks/
│   └── lab1_heart_disease.ipynb
├── reports/
│   └── metrics.json
├── src/
│   ├── data/make_dataset.py       # загрузка + искусственные пропуски
│   ├── features/transformers.py   # AgeSquaredTransformer
│   └── models/train.py            # пайплайн + GridSearchCV
├── .dvc/
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
# окружение (уже есть .venv_heart, либо создайте своё)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# данные
python -m src.data.make_dataset

# DVC (если ещё не инициализирован)
dvc init
dvc add data/raw/heart_disease.csv
git add data/raw/heart_disease.csv.dvc data/raw/.gitignore .dvc .dvcignore

# обучение
python -m src.models.train

# ноутбук
jupyter lab notebooks/lab1_heart_disease.ipynb
```

## Что сделано по шагам задания

1. **Структура каталогов** — `data/raw`, `data/processed`, `notebooks`, `src`, `models`, `reports`.
2. **Git** — репозиторий инициализирован, `.gitignore` настроен.
3. **Виртуальное окружение** — зависимости зафиксированы в `requirements.txt` (pandas, scikit-learn, matplotlib, seaborn, dvc, jupyterlab, …).
4. **Датасет** — Heart Disease сохранён в `data/raw/heart_disease.csv`. Пропусков не было → искусственно внесено ~10% NaN в числовые колонки `chol` и `trestbps`.
5. **DVC** — `dvc init`, `dvc add data/raw/heart_disease.csv`.
6. **Пайплайн**:
   - split 80/20, `random_state=42`, `stratify=y`;
   - `AgeSquaredTransformer` (`BaseEstimator` + `TransformerMixin`);
   - `ColumnTransformer`: числовые → медиана + `StandardScaler`; категориальные → мода + `OneHotEncoder(handle_unknown='ignore')`;
   - модель: `LogisticRegression` + `GridSearchCV` по `C`.
7. **Метрики** (тест, n=61) — см. ниже.
8. **Модель** — `models/heart_disease_pipeline.joblib`.
9. **Отчёт** — этот README + `reports/metrics.json`.

## Признаки

- **Числовые:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `age_squared` (создаётся трансформером)
- **Категориальные:** `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`

## Результаты

| Метрика | Значение |
|---------|----------|
| Лучший `C` (GridSearchCV) | `0.1` |
| CV ROC-AUC (5-fold) | **0.919** |
| Test Accuracy | **0.869** |
| Test ROC-AUC | **0.904** |

Выводы:

- Пайплайн с импутацией и масштабированием корректно обрабатывает искусственные пропуски.
- Признак `age²` встроен в конвейер до `ColumnTransformer`, поэтому применяется и на train, и на test без утечки логики.
- Логистическая регрессия с `C=0.1` даёт устойчивый ROC-AUC ~0.90 на отложенной выборке.

## DVC

```bash
dvc status
# Data and pipelines are up to date.
```

Данные лежат в локальном кэше DVC; в git коммитятся только `.dvc`-метаданные.

## Воспроизведение метрик

```bash
python -m src.models.train
cat reports/metrics.json
```
