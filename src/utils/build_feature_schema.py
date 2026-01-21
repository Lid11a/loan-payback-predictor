import json
from pathlib import Path

import pandas as pd


CATEGORICAL = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade",
]

NUMERIC = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
]


def build_schema_from_csv(csv_path: str, out_path: str = "models/feature_schema.json") -> None:
    df = pd.read_csv(csv_path)

    categorical_values = {}
    for col in CATEGORICAL:
        vals = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        categorical_values[col] = sorted(vals)

    schema = {
        "features_expected": len(CATEGORICAL) + len(NUMERIC),
        "categorical": categorical_values,
        "numeric": {col: {"type": "number"} for col in NUMERIC},
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_schema_from_csv("data/raw/train.csv")
    print("Saved schema to models/feature_schema.json")
