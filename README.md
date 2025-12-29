![CI](https://github.com/lid11a/loan-payback-predictor/actions/workflows/ci.yml/badge.svg)

# Loan Payback Prediction

ML-проект по бинарной классификации: по данным клиента предсказать, **вернёт ли он кредит**.

Проект сделан как портфолио "от анализа до сервиса":
- **ноутбук (EDA + эксперименты)** — основной объём работы: анализ данных, сравнение моделей, фичи, пороги, интерпретация, исследовательские блоки;
- **production-код** — обучение/предсказания, MLflow, тесты, FastAPI, Docker, CI.

---

## Что внутри и в чём "прикол" проекта

### 1) Notebook = исследование и выбор лучшего решения (основа проекта)

В `notebooks/loan_payback_predictor.ipynb` собрана основная аналитическая работа:

- EDA: первичный анализ данных, распределения, пропуски/аномалии, базовые закономерности;

- Исследовательская кластеризация: попытка сегментации клиентов для лучшего понимания структуры данных;

- Сравнение моделей и пайплайнов:

    - выбранные признаки vs. полный набор,

    - разные варианты препроцессинга (в т.ч. скейлеры),

    - разные подходы к подготовке данных (включая лог-трансформации числовых признаков и обработку выбросов);

- выбор рабочей точки модели: подбор порога классификации под задачу и внедрение early stopping для лучшей модели;

- интерпретация: важность признаков и выводы о том, какие факторы сильнее всего влияют на вероятность возврата кредита.

Итог ноутбука: выбрана “лучшая” модель и логика, которые затем перенесены в `src/` как воспроизводимый код.

### 2) Code = воспроизводимый pipeline + сервис
Код в `src/` делает то же самое “по кнопке”, без ручных действий:
- `train.py` обучает финальную модель и сохраняет bundle,
- `predict.py` делает предсказания на test.csv и формирует файлы,
- FastAPI отдаёт инференс через HTTP,
- MLflow логирует эксперименты,
- pytest защищает ключевую логику,
- Docker позволяет запустить проект в одинаковом окружении.

---


## Структура проекта

```
loan_py/
├── src/
│ ├── api/ #
│ │ └── app.py # FastAPI: /predict, /predict_batch, /ready
│ ├── data/ # загрузка/чтение данных
│ │ ├── download.py # Kaggle CLI download + распаковка
│ │ ├── load.py # чтение train.csv / test.csv
│ │ └── preprocessing.py # сбор X/y + OHE препроцессор
│ ├── models/
│ │ ├── predict.py # загрузка bundle + предсказания + сохранение файлов
│ │ ├── promote.py # промоут MLflow run → models/best_model.joblib
│ │ ├── train.py # обучение + подбор порога по target FPR + сохранение bundle
│ │ ├── train_baseline.py #
│ │ └── train_holdout.py #
│ └── utils/
│ ├── config.py # константы (имена колонок, slug соревнования и т.д.)
│ └── logger.py # логирование
├── tests/ # pytest-тесты
│ ├── test_conftest.py #
│ ├── test_api.py #
│ ├── test_download.py #
│ ├── test_load.py #
│ ├── test_logger_setup.py #
│ ├── test_predict_cli.py #
│ ├── test_preprocessing.py #
│ ├── test_threshold.py #
│ └── test_utils.py #
├── notebooks/ # ноутбук с EDA/экспериментами
│ └──  loan_payback_predictor.ipynb
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── README.md
├── requirements-all.txt
└── requirements-ci.txt
```

---

## Данные Kaggle

Данные скачиваются из Kaggle соревнования через CLI.

Нужно настроить Kaggle API token:
1. Kaggle → Account → Create New API Token
2. положить `kaggle.json` в:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

По умолчанию данные будут в:
- `data/raw/train.csv`
- `data/raw/test.csv`

---

## Команды

### 0) Проверка что вы в корне проекта
Вы должны находиться в папке, где лежат `src/`, `tests/`, `Dockerfile`.

---

### 1) Установка зависимостей

**Windows (PowerShell)**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-all.txt
```

**macOS / Linux**

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-all.txt
```

### 2) Запуск тестов

```
pytest
```

### 3) Обучение модели

```
python src/models/train.py
```

Что происходит при обучении:

- данные скачиваются (если их ещё нет локально),

- строится препроцессинг (OneHotEncoder для категориальных),

- обучается модель LightGBM,

- подбирается порог классификации под целевой FPR (по умолчанию ```target_fpr = 0.20```),

- сохраняется bundle в ```models/best_model.joblib``` (модель + препроцессор + threshold + meta).


### 4) Предсказание

```
python src/models/predict.py
```

Что получается на выходе:

- ```data/predictions/submission_model.csv``` — вероятности (для сабмита)

- ```data/predictions/decisions_model_thr_XXXX.csv``` — вероятности + бинарное решение по порогу

### 5) Запуск API локально (без Docker)

```
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**Swagger:**

http://127.0.0.1:8000/docs

**Проверка, что модель загрузилась:**

http://127.0.0.1:8000/ready

---

### MLflow

При обучении `train.py` логирует параметры, метрики и артефакты в MLflow.

**Запуск UI:**

```
mlflow ui
```

**Открыть:**

http://127.0.0.1:5000

---

### API эндпоинты

- `GET /health` — жив ли процесс

- `GET /ready` — загружена ли модель (bundle)

- `GET /features` — список ожидаемых фичей

- `POST /predict` — 1 запись

- `POST /predict_batch` — много записей

**Swagger:**

http://127.0.0.1:8000/docs

---

### Docker (воспроизводимый запуск)

1) Собрать образ

```
docker build -t loan-api:repro .
```

2) Запустить контейнер

Папка `models/` не хранится в репозитории, поэтому её нужно “подмонтировать” в контейнер.

**Windows (PowerShell)**

```
$models=(Resolve-Path .\models).Path; docker run --rm -p 8000:8000 -v "${models}:/app/models" loan-api:repro
```

**macOS / Linux**

```
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" loan-api:repro
```

**Swagger:**

http://127.0.0.1:8000/docs

---

### Пример запроса в API

Сначала посмотри ожидаемые фичи:

- `GET /features`

Пример:

```
POST /predict
{
  "features": {
    "gender": "Female",
    "marital_status": "Single",
    "education_level": "Bachelor",
    "employment_status": "Employed",
    "loan_purpose": "Debt Consolidation",
    "grade_subgrade": "B2",
    "annual_income": 50000,
    "debt_to_income_ratio": 0.25,
    "credit_score": 680,
    "loan_amount": 12000,
    "interest_rate": 12.5
  }
}
```

---

### CI (GitHub Actions)

При push/PR:

- ставятся зависимости из requirements-ci.txt,

- запускаются тесты,

- считается coverage.

---

## Используемые технологии
### Основной пайплайн (код в ```src/```)

- Python

- pandas, numpy

- scikit-learn (ColumnTransformer, OneHotEncoder, train/test split, метрики)

- LightGBM

- joblib

- pytest + coverage

- logging

- Kaggle CLI

- MLflow

- FastAPI + Uvicorn

- Docker

### Эксперименты в ноутбуке (EDA/пробы моделей)

- scipy

- matplotlib, seaborn, plotly

- xgboost

- catboost

- (в ноутбуке использовались и другие модели sklearn: LogisticRegression, RandomForest, SVC и т.д.)
