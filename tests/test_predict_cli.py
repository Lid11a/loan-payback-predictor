# tests/test_predict_cli.py
from src.models.predict import resolve_model_path


def test_resolve_model_path_default():
    p = resolve_model_path("models/best_model.joblib", run_id=None, experiment_id="1")
    assert str(p).replace("\\", "/").endswith("models/best_model.joblib")


def test_resolve_model_path_with_run_id():
    p = resolve_model_path("models/best_model.joblib", run_id="ABC", experiment_id="1")
    s = str(p).replace("\\", "/")
    assert s.endswith("mlruns/1/ABC/artifacts/model/best_model.joblib")
