# Loan Payback Prediction

Проект по бинарной классификации: по данным клиента предсказать, вернёт ли он кредит.

Сделано как учебный проект с упором на:
- понятную структуру кода,
- воспроизводимый запуск,
- тесты,
- аккуратную инженерную подачу (Git, requirements, README).

---

## Структура проекта

```
loan_py/
├── src/
│ ├── data/ # загрузка/чтение данных
│ │ ├── download.py # Kaggle CLI download + распаковка
│ │ ├── load.py # чтение train.csv / test.csv
│ │ └── preprocessing.py # сбор X/y + OHE препроцессор
│ ├── models/
│ │ ├── train.py # обучение + подбор порога по target FPR + сохранение bundle
│ │ ├── train_baseline.py #
│ │ ├── train_holdout.py #
│ │ └── predict.py # загрузка bundle + предсказания + сохранение файлов
│ └── utils/
│ ├── config.py # константы (имена колонок, slug соревнования и т.д.)
│ └── logger.py # логирование
├── tests/ # pytest-тесты
│ ├── test_download.py #
│ ├── test_preprocessing.py #
│ ├── test_threshold.py #
│ └── test_utils.py #
├── notebooks/ # ноутбук с EDA/экспериментами
│ └──  loan_payback_predictor.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Установка

Рекомендуется виртуальное окружение.

### Windows (PowerShell)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Данные (Kaggle)

Данные берутся из Kaggle соревнования через Kaggle API (CLI).

Важно:

- файлы данных (data/) не хранятся в репозитории (они скачиваются при запуске),

- нужно, чтобы у тебя был настроен Kaggle API (token).

По умолчанию данные кладутся в:

- data/raw/train.csv

- data/raw/test.csv

---

### Обучение модели

Запуск обучения (из корня проекта):

```
python src/models/train.py
```

Что происходит при обучении:

- данные скачиваются (если их ещё нет локально),

- строится препроцессинг (OneHotEncoder для категориальных),

- обучается модель LightGBM,

- подбирается порог классификации под целевой FPR (по умолчанию ```target_fpr=0.20```),

- сохраняется bundle в ```models/best_model.joblib``` (модель + препроцессор + threshold + meta).

---

### Предсказание

Запуск предсказаний (из корня проекта):

```
python src/models/predict.py
```

Что получается на выходе:

- ```data/predictions/submission_model.csv``` — вероятности (для сабмита)

- ```data/predictions/decisions_model_thr_XXXX.csv``` — вероятности + бинарное решение по порогу

---

### Тесты

Запуск тестов:

```
pytest
```

Тестами покрыты ключевые части:

- preprocessing,

- подбор порога по целевому FPR,

- скачивание данных (через monkeypatch).

---

### Ноутбук (EDA / эксперименты)

В ```notebooks/``` лежит ноутбук с исследованием данных и экспериментами.
Чекпоинты Jupyter (```notebooks/.ipynb_checkpoints/```) в Git не добавляются.

---

### Используемые технологии
#### Основной пайплайн (код в ```src/```)

- Python

- pandas, numpy

- scikit-learn (ColumnTransformer, OneHotEncoder, train/test split, метрики)

- LightGBM

- joblib

- pytest

- logging

- Kaggle CLI

#### Эксперименты в ноутбуке (EDA/пробы моделей)

- scipy

- matplotlib, seaborn, plotly

- xgboost

- catboost

- (в ноутбуке использовались и другие модели sklearn: LogisticRegression, RandomForest, SVC и т.д.)
