# tests/test_predict.py
from src.models.predict import resolve_model_path


def test_resolve_model_path_default():
    """
    The function verifies that `resolve_model_path` returns the default model path
    when no run_id is provided.
    """
    p = resolve_model_path("models/best_model.joblib", run_id = None, experiment_id = "1")
    assert str(p).replace("\\", "/").endswith("models/best_model.joblib")


def test_resolve_model_path_with_run_id():
    """
    The function verifies that `resolve_model_path` builds an MLflow artifact path
    when a run_id is provided.
    """
    p = resolve_model_path("models/best_model.joblib", run_id = "ABC", experiment_id = "1")
    s = str(p).replace("\\", "/")
    assert s.endswith("mlruns/1/ABC/artifacts/model/best_model.joblib")

