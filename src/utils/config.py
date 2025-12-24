# src/utils/config.py
from __future__ import annotations

COMPETITION_SLUG = "playground-series-s5e11"
PROJECT_NAME = "loan-payback-predictor"

TARGET_COL = "loan_paid_back"
ID_COL = "id"

NUM_LOG_FEATURES = ["annual_income", "debt_to_income_ratio"]

SELECTED_EXCLUDE_NUM_LOG = ["annual_income"]
SELECTED_EXCLUDE_NUM_NON_LOG = ["loan_amount"]
SELECTED_EXCLUDE_CATEGORICAL = ["marital_status"]
