# src/monitoring/drift.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.load import load_kaggle_data
from src.data.preprocessing import make_xy, split_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


# PSI utilities

_EPS = 1e-12


def _safe_ratio(a: float, b: float) -> float:
    """
    Compute a / b in a numerically safe way.

    A small epsilon is added to the denominator to avoid
    division by zero and unstable values when b is close to zero.
    """
    return float(a) / float(b + _EPS)


def _psi_from_distributions(ref: np.ndarray, cur: np.ndarray) -> float:
    """
    PSI = sum( (cur - ref) * ln(cur/ref) )
    Both ref and cur are arrays of proportions (sum ~ 1).
    """
    ref = np.asarray(ref, dtype = float)
    cur = np.asarray(cur, dtype = float)

    # The function clips distributions to avoid log(0) and division by zero issues
    ref = np.clip(ref, _EPS, 1.0)
    cur = np.clip(cur, _EPS, 1.0)

    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(
    ref: pd.Series,
    cur: pd.Series,
    n_bins: int = 10,
    include_missing_bin: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    PSI for numeric feature using quantile bins computed on reference.

    Steps:
    - compute quantile edges on reference (dropna)
    - cut both ref and cur using the same edges
    - count proportions per bin
    - optional separate bin for missing values

    Returns:
      psi_value, debug_info (bin_count_ref, bin_count_cur, ...)
    """
    ref_s = pd.to_numeric(ref, errors = "coerce")
    cur_s = pd.to_numeric(cur, errors = "coerce")

    ref_non = ref_s.dropna()
    if ref_non.empty:
        # If reference is entirely missing/unusable, PSI is not meaningful.
        # Return 0, but log it as a warning in report layer if needed.
        info = {"ref_non_missing": 0.0, "cur_non_missing": float(cur_s.notna().mean())}
        return 0.0, info

    # Quantile edges (on reference only)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref_non.values, qs)

    # The function deduplicates quantile edges to guarantee strictly increasing bin borders
    edges = np.unique(edges)
    if edges.size < 3:
        # Almost constant feature: treat as no drift (PSI ~ difference in missing only)
        ref_missing = float(ref_s.isna().mean())
        cur_missing = float(cur_s.isna().mean())
        # "missing-only psi": two bins (missing / not missing)
        if include_missing_bin:
            ref_dist = np.array([ref_missing, 1.0 - ref_missing])
            cur_dist = np.array([cur_missing, 1.0 - cur_missing])
            return _psi_from_distributions(ref_dist, cur_dist), {
                "ref_missing_rate": ref_missing,
                "cur_missing_rate": cur_missing,
                "note": "constant_or_low_variance",
            }
        return 0.0, {"note": "constant_or_low_variance"}

    # Build bins with open ends
    bins = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    ref_bins = pd.cut(ref_s, bins = bins, include_lowest = True)
    cur_bins = pd.cut(cur_s, bins = bins, include_lowest = True)

    ref_counts = ref_bins.value_counts(dropna = False).sort_index()
    cur_counts = cur_bins.value_counts(dropna = False).sort_index()

    # Align indexes
    all_idx = ref_counts.index.union(cur_counts.index)
    ref_counts = ref_counts.reindex(all_idx, fill_value = 0)
    cur_counts = cur_counts.reindex(all_idx, fill_value = 0)

    ref_dist = (ref_counts / max(len(ref_s), 1)).values
    cur_dist = (cur_counts / max(len(cur_s), 1)).values

    if not include_missing_bin:
        # Remove NaN bin produced by pd.cut for missing values (if any)
        # In value_counts(dropna = False) NaN is a key; for CategoricalIndex it may appear as NaN.
        # Easiest: mask by notna on index
        mask = pd.notna(all_idx)
        ref_dist = ref_dist[mask]
        cur_dist = cur_dist[mask]

        # renormalize
        ref_dist = ref_dist / max(ref_dist.sum(), _EPS)
        cur_dist = cur_dist / max(cur_dist.sum(), _EPS)

    psi = _psi_from_distributions(ref_dist, cur_dist)

    info = {
        "ref_missing_rate": float(ref_s.isna().mean()),
        "cur_missing_rate": float(cur_s.isna().mean()),
        "n_bins_used": float(len(ref_dist)),
    }
    return psi, info


def psi_categorical(
    ref: pd.Series,
    cur: pd.Series,
    include_missing_as_category: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    PSI for categorical feature using category proportions.

    - We build a set of categories from BOTH ref and cur (union)
      so "new categories" in current are included.
    - Missing can be treated as its own category.
    """
    ref_s = ref.astype("object")
    cur_s = cur.astype("object")

    if include_missing_as_category:
        ref_s = ref_s.where(ref_s.notna(), "__MISSING__")
        cur_s = cur_s.where(cur_s.notna(), "__MISSING__")

    ref_counts = ref_s.value_counts(dropna = False)
    cur_counts = cur_s.value_counts(dropna = False)

    cats = ref_counts.index.union(cur_counts.index)
    ref_counts = ref_counts.reindex(cats, fill_value = 0)
    cur_counts = cur_counts.reindex(cats, fill_value = 0)

    ref_dist = (ref_counts / max(len(ref_s), 1)).values
    cur_dist = (cur_counts / max(len(cur_s), 1)).values

    psi = _psi_from_distributions(ref_dist, cur_dist)

    info = {
        "ref_missing_rate": float(pd.isna(ref).mean()),
        "cur_missing_rate": float(pd.isna(cur).mean()),
        "ref_unique": float(ref_s.nunique(dropna = False)),
        "cur_unique": float(cur_s.nunique(dropna = False)),
    }
    return psi, info


# Report building

@dataclass(frozen = True)
class DriftThresholds:
    ok: float = 0.10
    warn: float = 0.25  # >= warn => DRIFT


def psi_status(psi_value: float, thr: DriftThresholds) -> str:
    """
    Map a PSI value to a categorical drift status.

    The status is determined using predefined PSI thresholds:
        - OK:    negligible or no distribution shift
        - WARN:  moderate shift that may require attention
        - DRIFT: significant shift indicating potential data instability
    """
    if psi_value < thr.ok:
        return "OK"
    if psi_value < thr.warn:
        return "WARN"
    return "DRIFT"


def build_feature_drift_report(
    x_ref: pd.DataFrame,
    x_cur: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    n_bins_numeric: int = 10,
    thresholds: DriftThresholds = DriftThresholds(),
) -> pd.DataFrame:
    """
    Build a per-feature drift report (PSI-based).
    Works on raw features (before OHE).
    """
    rows: List[Dict[str, object]] = []

    # Ensure same columns exist
    # (Your pipeline already enforces consistent columns between train/test; still safe.)
    for col in numeric_features:
        if col not in x_ref.columns or col not in x_cur.columns:
            continue
        psi, info = psi_numeric(x_ref[col], x_cur[col], n_bins = n_bins_numeric, include_missing_bin = True)
        rows.append(
            {
                "feature": col,
                "type": "numeric",
                "psi": psi,
                "status": psi_status(psi, thresholds),
                "ref_missing_rate": info.get("ref_missing_rate", np.nan),
                "cur_missing_rate": info.get("cur_missing_rate", np.nan),
                "ref_unique": np.nan,
                "cur_unique": np.nan,
            }
        )

    for col in categorical_features:
        if col not in x_ref.columns or col not in x_cur.columns:
            continue
        psi, info = psi_categorical(x_ref[col], x_cur[col], include_missing_as_category = True)
        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "psi": psi,
                "status": psi_status(psi, thresholds),
                "ref_missing_rate": info.get("ref_missing_rate", np.nan),
                "cur_missing_rate": info.get("cur_missing_rate", np.nan),
                "ref_unique": info.get("ref_unique", np.nan),
                "cur_unique": info.get("cur_unique", np.nan),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sort by most severe drift first
    order = pd.CategoricalDtype(categories = ["DRIFT", "WARN", "OK"], ordered = True)
    df["status"] = df["status"].astype(order)
    df = df.sort_values(["status", "psi"], ascending = [True, False]).reset_index(drop = True)

    # Nice rounding for readability (but keep full float in CSV? usually OK to round)
    df["psi"] = df["psi"].astype(float)
    df["ref_missing_rate"] = df["ref_missing_rate"].astype(float)
    df["cur_missing_rate"] = df["cur_missing_rate"].astype(float)

    return df


def run_offline_drift_monitoring(
    data_dir: str | Path = "data/raw",
    out_dir: str | Path = "reports",
    out_name: str = "drift_report.csv",
    n_bins_numeric: int = 10,
    thresholds: DriftThresholds = DriftThresholds(),
) -> Path:
    """
    End-to-end offline drift report:
    - loads Kaggle train/test
    - builds X_train (reference) and X_test (current)
    - computes PSI feature drift report
    - saves it to reports/drift_report.csv

    Returns path to the saved report.
    """
    logger.info("Drift monitoring started. data_dir = %s", data_dir)

    train_df, test_df = load_kaggle_data(data_dir)
    x_train, _, x_test = make_xy(train_df, test_df)

    spec = split_features(train_df)

    logger.info(
        "Drift inputs prepared. x_ref = %s x_cur = %s numeric = %s categorical = %s",
        x_train.shape,
        x_test.shape,
        len(spec.numeric),
        len(spec.categorical),
    )

    report = build_feature_drift_report(
        x_ref = x_train,
        x_cur = x_test,
        numeric_features = spec.numeric,
        categorical_features = spec.categorical,
        n_bins_numeric = n_bins_numeric,
        thresholds = thresholds,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents = True, exist_ok = True)

    report_path = out_path / out_name
    report.to_csv(report_path, index = False)

    # Simple summary for logs
    if not report.empty:
        counts = report["status"].value_counts().to_dict()
        logger.info("Drift report saved: %s status_counts = %s", report_path, counts)
        logger.info("Top-5 by PSI:\n%s", report.head(5).to_string(index = False))
    else:
        logger.warning("Drift report is empty (no features found?). Saved: %s", report_path)

    return report_path


if __name__ == "__main__":
    p = run_offline_drift_monitoring()
    logger.info("Done. Report: %s", p)
