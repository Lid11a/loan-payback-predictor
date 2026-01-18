![CI](https://github.com/lid11a/loan-payback-predictor/actions/workflows/ci.yml/badge.svg)

# Loan Payback Prediction

A binary classification machine learning project focused on predicting **whether a client will repay a loan** 
based on their profile and financial characteristics.

This repository demonstrates the **full lifecycle of an ML project**:

1) in-depth exploratory data analysis and model experimentation in a Jupyter Notebook;  
2) selection of a final solution under explicit business constraints (target FPR);  
3) transition from research results to a reproducible, production-style ML service.

The project includes reproducible training and inference, MLflow experiment tracking, automated tests, CI,
a FastAPI-based inference API, Dockerized deployment, and offline feature drift monitoring (PSI).

**Languages:** [English](README.md) | [Русский](README_RU.md)

---

## Table of contents

- [Project Overview](#project-overview)
- [Exploratory Analysis](#exploratory-analysis)
- [Code Implementation](#code-implementation)
- [Feature Drift Monitoring (PSI)](#feature-drift-monitoring-psi)
- [Project Structure](#project-structure)
- [Data](#data)
- [Running the Project](#running-the-project)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Project overview

This project focuses on a **credit scoring problem** and demonstrates the complete workflow 
of a binary classification task: from exploratory data analysis and model experimentation 
to a reproducible engineering implementation and online inference.

The goal of the project is to demonstrate the model selection process and the transition 
from research insights to an engineering-oriented ML solution suitable for practical use and further development.

The project includes:

- exploratory data analysis, including analysis of feature distributions, target variable behavior, 
  and the structure of factors influencing loan default risk;
- statistical analysis and hypothesis testing used to interpret data patterns
  and compare statistical findings with model behavior;
- experimental comparison of different model families and training approaches;
- selection of a final solution under an explicit business constraint
  (classification threshold selection based on a target FPR);
- implementation of a reproducible ML pipeline
  with shared preprocessing and strict train/inference consistency;
- implementation of training and inference workflows with experiment tracking via MLflow;
- implementation of an HTTP API for online inference based on FastAPI;
- setup of automated testing and CI to ensure code correctness;
- implementation of reproducible service execution using Docker;
- implementation of offline feature drift monitoring (PSI) to track input data stability.

The project is presented in a structured manner, covering the exploratory phase,
the production-style implementation, and the supporting components that ensure reproducibility and quality control.

---

## Exploratory analysis

The exploratory part of the project is implemented in `notebooks/loan_payback_predictor.ipynb`
and serves as the foundation for all subsequent modeling and pipeline design decisions.

The notebook follows a structured approach to data analysis and modeling.

### Exploratory data analysis (EDA)

Initial analysis focuses on understanding the dataset structure,
feature properties, and target variable behavior.

EDA includes:

- inspection of dataset structure, feature types, and target distribution;
- analysis of numerical feature distributions and outlier detection;
- missing value analysis and assessment of their potential impact;
- categorical feature analysis, including:
  - category distributions;
  - default rates across categories.

### Statistical analysis

Statistical analysis is used to quantitatively assess relationships between features and the target
and to validate patterns identified during EDA.

This includes:

- statistical tests for dependency between categorical features and the target;
- interpretation of statistical significance;
- comparison of statistical relevance with model-based feature importance.

### Data structure exploration

Clustering is applied as an exploratory technique to analyze the structure of the customer population.

**Methods applied:**

- KMeans;
- Gaussian Mixture Models (GMM).

**Clustering analysis includes:**

- selection of the number of clusters using the silhouette score;
- analysis of target distribution within clusters;
- visualization of cluster structure in reduced dimensionality using PCA;
- analysis of `grade_subgrade` distribution across clusters.

Clustering is applied for analytical purposes only and is excluded from the final ML pipeline.

### Model experiments

Model experiments aim to compare different algorithm families
in terms of predictive performance, stability, and feature sensitivity.

The following approaches are evaluated:

- linear models;
- distance-based and margin-based methods (KNN, SVM);
- probabilistic models (Naive Bayes);
- decision trees and ensemble methods;
- neural networks (MLP).

Experiments are conducted with different preprocessing strategies and feature sets
to assess model robustness and identify the most suitable approach for the task.

### Final model selection

The final solution is selected based on experimental results
and detailed analysis of model behavior.

Key steps include:

- application of early stopping to control overfitting;
- feature importance analysis and interpretation of results;
- classification threshold selection under a predefined business constraint (target FPR).

---

## Code implementation

The codebase translates research findings into a reproducible ML pipeline and inference service.

The primary objective of the implementation is to ensure:
- consistent data processing between training and inference;
- reproducibility of experiments;
- practical usability of the selected solution.

### Training and inference

Training and inference are implemented as separate workflows sharing a unified data processing pipeline.

During training:

- a single preprocessing pipeline is constructed;
- the final model is trained with overfitting control (early stopping);
- a classification threshold is selected based on the target FPR;
- a unified model bundle is saved, including the model, preprocessor, threshold, and auxiliary metadata.

Inference uses the saved bundle directly, eliminating discrepancies between training and prediction logic.

### Reproducibility and logging

Two complementary mechanisms are used to ensure reproducibility and observability of the system.

Standard Python logging is applied to track pipeline execution and service logic.
Logs are generated across the core project modules and persisted to files,
enabling detailed analysis of pipeline behavior and effective error diagnosis.

MLflow is used to track training runs. Each run stores training parameters, quality metrics, and model artifacts,
ensuring experiment reproducibility and enabling systematic comparison of different runs.

### Online inference service

An HTTP API for online inference is implemented using FastAPI.

The service:

- loads the saved model bundle at startup;
- validates incoming request schemas;
- returns prediction probabilities and binary decisions based on the selected classification threshold.


### Testing and quality control

Key components of the ML pipeline and service are covered by automated tests (pytest).

Tests validate:

- data download and loading (`test_download.py`, `test_load.py`);
- preprocessing and feature preparation (`test_preprocessing.py`);
- threshold selection and application (`test_threshold.py`);
- CLI inference workflows (`test_predict_cli.py`);
- API endpoint behavior (`test_api.py`);
- feature drift monitoring logic (`test_drift.py`);
- logging setup and utility functions (`test_logger_setup.py`, `test_utils.py`).

Tests are executed automatically in the CI pipeline, reducing the risk of regressions.

### Reproducible execution

Docker is used to ensure an isolated and reproducible runtime environment.

The Docker image contains all required dependencies and allows the inference service to be run 
in a consistent environment, simplifying validation and demonstration.

---

## Feature drift monitoring (PSI)

The project includes a dedicated offline feature drift monitoring module 
designed to assess post-training input data stability.

Drift monitoring evaluates changes in feature distributions in new data relative to the training (reference) dataset
and is implemented as an independent batch process.

The **Population Stability Index (PSI)** is used as the drift metric.
Comparisons are performed between the reference dataset and a new data batch.

Drift is calculated at the raw feature level (before one-hot encoding), improving interpretability.

For numerical features, PSI is computed using quantile-based bins derived from the reference data, 
with a separate bin for missing values.
For categorical features, category proportions are used, including previously unseen values.

For each feature, an aggregated report is generated containing:
- PSI value;
- drift status (`OK / WARN / DRIFT`);
- auxiliary statistics for missing values and feature cardinality.

The module is intended for analytical monitoring and is not part of the online inference path.

**Run drift monitoring:**

```
python -m src.monitoring.drift
```

After execution of the command, a CSV report is generated:

```
reports/drift_report.csv
```

---

## Project structure

The repository structure is shown below and reflects the separation of the project into the exploratory component,
the ML pipeline implementation, and the service components.

```
loan_py/
├── src/
│ ├── api/                      # HTTP API for online inference
│ │ └── app.py                  # FastAPI application: /predict, /predict_batch, /ready
│ ├── data/                     # Raw data download and preparation
│ │ ├── download.py             # Dataset download from Kaggle via CLI
│ │ ├── load.py                 # Loading train.csv / test.csv
│ │ └── preprocessing.py        # X/y construction and preprocessing (OHE)
│ ├── models/                   # Model training, inference, and management
│ │ ├── predict.py              # Inference using the saved bundle and prediction export
│ │ ├── promote.py              # Promotion of an MLflow run to the final model
│ │ ├── train.py                # Final model training and threshold selection
│ │ ├── train_baseline.py       # Baseline model for comparison
│ │ └── train_holdout.py        # Alternative training scenario (holdout)
│ ├── monitoring/
│ │ └── drift.py                # Offline feature drift monitoring (PSI)
│ └── utils/
│   ├── config.py               # Project constants (column names, parameters, etc.)
│   └── logger.py               # Centralized logging configuration
├── tests/                      # Automated tests (pytest)
│ ├── test_conftest.py          # Shared test fixtures
│ ├── test_api.py               # API endpoint tests
│ ├── test_download.py          # Data download tests
│ ├── test_drift.py             # Drift monitoring module tests
│ ├── test_load.py              # Data loading tests
│ ├── test_logger_setup.py      # Logging configuration tests
│ ├── test_predict_cli.py       # CLI inference tests
│ ├── test_preprocessing.py     # Preprocessing tests
│ ├── test_threshold.py         # Threshold selection logic tests
│ └── test_utils.py             # Utility function tests
├── notebooks/                  # Exploratory analysis
│ └── loan_payback_predictor.ipynb  # EDA, experiments, final model selection
├── .dockerignore                # Docker build exclusions
├── .gitignore                   # Git exclusions
├── Dockerfile                   # Docker image for the API
├── pyproject.toml               # Project and tool configuration
├── README.md                    # Project description
├── LICENSE                      # Project license
├── requirements-all.txt         # Full dependency list
└── requirements-ci.txt          # CI dependencies
```

---

## Data

The project uses data from a Kaggle competition.
The data is not stored in the repository and is downloaded locally via the Kaggle API.

To use the project, the following steps are required:
1) cloning the repository;
2) configuring Kaggle API access;
3) downloading training and test datasets locally.

The complete sequence of steps is described below.

### 1) Project setup

Clone the repository and navigate to its root directory:

```
git clone https://github.com/lid11a/loan-payback-predictor.git
cd loan-payback-predictor
```

After executing the `cd` command, you are located in the project root directory,
which contains `src/`, `tests/`, `notebooks/`, `Dockerfile`, and `README`.

### 2) Kaggle data access configuration

The project uses the **Kaggle CLI** and a personal API token to download the dataset.

  1. Open the Kaggle website and navigate to your account settings: **Profile → Account**.

  2. In the **API** section, click **Create New API Token**. 
  A `kaggle.json` file containing your API credentials will be downloaded.

  3. Place the `kaggle.json` file in the default Kaggle directory:

    - **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
    - **Linux / macOS:** `~/.kaggle/kaggle.json`

  For Linux/macOS, file permissions must also be set:

  ```
  chmod 600 ~/.kaggle/kaggle.json
  ```

### 3) Data download and storage

After configuring Kaggle API access, the dataset is downloaded automatically
when model training is launched, or it can be downloaded manually in advance.

By default, the following files are created in the project after download:

- `data/raw/train.csv` — training dataset;
- `data/raw/test.csv` — test dataset.

If these files are already present locally, the download step is skipped.

---

## Running the project

This section describes the standard workflow for using the project, 
from dependency installation to model training and inference.
All commands assume that you are located in the repository root directory.

### 1) Dependency installation

The project uses Python and a virtual environment.
It is recommended to create a dedicated environment before installing dependencies.

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

### 2) Running tests

Before starting model training, it is recommended to verify the installation and basic project functionality.

```
pytest
```

### 3) Model training

Final model training is launched using the following command:

```
python -m src.models.train
```

During training:

- the dataset is automatically downloaded from Kaggle if not available locally;
- a unified data preprocessing pipeline is constructed;
- a LightGBM-based binary classifier is trained;
- the classification threshold is selected under the specified business constraint
- (target FPR, default `target_fpr = 0.20`);
- the final model bundle is saved to `models/best_model.joblib`
  (model, preprocessor, classification threshold, and metadata).

### 4) Offline inference

After model training, inference can be performed on the test dataset:

```
python -m src.models.predict
```

Results are saved to the `data/predictions/` directory and include:

- a file with prediction probabilities (`data/predictions/submission_model.csv`);

- a file with probabilities and binary decisions based on the selected threshold
(`data/predictions/decisions_model_thr_XXXX.csv`).

### 5) Running the API locally (without Docker)

An HTTP API for online inference is available based on FastAPI:

```
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**Swagger UI** is available at:

http://127.0.0.1:8000/docs

**Service readiness check** (whether the model is loaded):

http://127.0.0.1:8000/ready

### 6) MLflow — experiment management

MLflow is used during training to track and reproduce experiments.

For each training run, MLflow logs:

- model and training parameters;
- evaluation metrics;
- model-related artifacts.

To view experiment history and compare runs, the MLflow web interface can be launched:

```
mlflow ui
```

By default, the interface is available at:

http://127.0.0.1:5000

### 7) API endpoints

The HTTP API is designed for online model inference and is implemented using FastAPI.

Available endpoints:

- `GET /health` — service availability check;
- `GET /ready` — model readiness check;
- `GET /features` — list of expected input features;
- `POST /predict` — inference for a single record;
- `POST /predict_batch` — inference for a batch of records.

Interactive documentation is available via **Swagger UI**:

http://127.0.0.1:8000/docs

#### Example API request

Before sending a request, the expected input features can be inspected via:

- `GET /features`

Example request for single-record inference:

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

### 8) Docker — reproducible service execution

A Docker environment is provided for isolated and reproducible service execution.

Build the Docker image:

```
docker build -t loan-api:repro .
```

Run the container:

The models/ directory is not stored in the repository and must be mounted into the container
to provide the service with access to the trained model.

**Windows (PowerShell)**

Run the following two commands:

```
$models = (Resolve-Path .\models).Path
docker run --rm -p 8000:8000 -v "${models}:/app/models" loan-api:repro
```

**macOS / Linux**

```
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" loan-api:repro
```

After startup, the API and Swagger UI are available at the same addresses as during local execution without Docker:

http://127.0.0.1:8000/docs

## 9) CI (GitHub Actions)

A CI pipeline based on **GitHub Actions** is configured for the repository.

The pipeline is automatically triggered on **push** and **pull request** events and performs the following steps:

- dependency installation (from `requirements-ci.txt`);
- test execution (`pytest`);
- test coverage computation.

The pipeline status is displayed in the badge at the top of the README.

---

## Technologies used

The project employs a toolchain covering the full ML workflow,
from exploratory data analysis to online inference and automated code validation.

### ML pipeline and service layer (`src/`)

- **Language and core libraries**  
  Python, pandas, numpy — data processing and numerical computation.

- **Machine learning and preprocessing**  
  scikit-learn — data processing pipelines, feature encoding, dataset splitting, and evaluation metrics.

- **Final model**  
  LightGBM — gradient boosting for binary classification; training with cross-validation and early stopping,
  handling of class imbalance.

- **Model persistence and artifacts**  
  joblib — storage of a unified model bundle (model, preprocessor, classification threshold, metadata).

- **Logging and experiment tracking**  
  logging — pipeline and service execution logging;
  MLflow — tracking of parameters, metrics, and training artifacts.

- **Testing and quality control**  
  pytest, coverage / pytest-cov — automated testing of core logic and test coverage monitoring.

- **Data acquisition**  
  Kaggle CLI — automated dataset download.

- **Inference service**  
  FastAPI, Uvicorn — HTTP API for online predictions.

- **Reproducibility and automation**  
  Docker — isolated and reproducible service execution;
  GitHub Actions — CI pipeline for automated code validation.

---

### Exploratory analysis and experiments (`notebooks/`)

The exploratory component applies multiple data analysis techniques and algorithm classes
to investigate data structure, test hypotheses, and select the final solution.

- **Data analysis and statistics**  
  pandas, numpy, scipy — analysis of feature distributions, missing values, and target variable behavior, 
  including statistical dependency tests (e.g., chi-square tests).

- **Visualization and EDA**  
  matplotlib, seaborn, plotly — visual analysis of distributions and class segmentation.

- **Clustering (exploratory analysis)**  
  KMeans, Gaussian Mixture Models (GMM) — investigation of data structure, 
  selection of the number of clusters using silhouette score,
  PCA visualization and analysis of target distribution within clusters.

- **Linear models**  
  Logistic Regression.

- **Distance- and margin-based methods**  
  K-Nearest Neighbors, Support Vector Machines.

- **Probabilistic models**  
  Naive Bayes.

- **Decision trees and ensemble methods**  
  Random Forest.

- **Gradient boosting**  
  XGBoost, CatBoost, LightGBM.

- **Neural networks**  
  Полносвязная нейронная сеть (MLP).

- **Experimentation environment**  
  Jupyter Notebook.

---

## License

This project is licensed under the MIT License.
