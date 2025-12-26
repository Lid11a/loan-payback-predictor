# tests/test_api.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient


class _FakePreprocessor:
    def transform(self, x: Any) -> Any:
        return x


class _FakeModel:
    def predict(self, x: Any) -> Any:
        # returns probability for each row
        n = len(x)
        return np.array([0.9] * n)


def _fake_bundle() -> Dict[str, Any]:
    return {
        "model": _FakeModel(),
        "preprocessor": _FakePreprocessor(),
        "threshold": 0.5,
        "meta": {
            "numeric_features": ["f1"],
            "categorical_features": ["c1"],
        },
    }


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """
    The fixture provides a TestClient with a patched model bundle.

    The API startup normally loads models/best_model.joblib from disk.
    In tests, this is replaced with a fake in-memory bundle to keep tests stable.
    """
    import src.api.app as app_module

    monkeypatch.setattr(app_module, "load_bundle", lambda *args, **kwargs: _fake_bundle())
    monkeypatch.setattr(app_module, "load_active_run_id", lambda *args, **kwargs: "RUN_TEST_123")

    # Ensure FEATURES is predictable
    monkeypatch.setattr(app_module, "get_expected_features", lambda bundle: ["c1", "f1"])

    with TestClient(app_module.app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready(client: TestClient) -> None:
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["model_loaded"] is True
    assert data["n_features_expected"] == 2
    assert data["active_run_id"] == "RUN_TEST_123"


def test_features(client: TestClient) -> None:
    r = client.get("/features")
    assert r.status_code == 200
    data = r.json()
    assert data["n_features_expected"] == 2
    assert data["features"] == ["c1", "f1"]


def test_predict_ok(client: TestClient) -> None:
    payload = {"features": {"c1": "A", "f1": 123}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["proba"] <= 1.0
    assert data["threshold"] == 0.5
    assert data["prediction"] in (0, 1)


def test_predict_invalid_feature_set(client: TestClient) -> None:
    payload = {"features": {"c1": "A"}}  # missing f1
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "missing_features" in detail
    assert "f1" in detail["missing_features"]


def test_predict_batch_ok(client: TestClient) -> None:
    payload = {
        "items": [
            {"c1": "A", "f1": 1},
            {"c1": "B", "f1": 2},
        ]
    }
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["threshold"] == 0.5
