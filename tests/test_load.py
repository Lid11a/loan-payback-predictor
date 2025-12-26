# tests/test_load.py
from pathlib import Path
import pandas as pd

from src.data.load import load_kaggle_data


def test_load_kaggle_data_reads_csvs(tmp_path, monkeypatch):
    data_dir = tmp_path / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # create fake csvs
    (data_dir / "train.csv").write_text("id,loan_paid_back,x\n1,0,10\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,x\n2,20\n", encoding="utf-8")

    # patch downloader: "pretend everything is downloaded"
    import src.data.load as load_module
    monkeypatch.setattr(load_module, "download_kaggle_competition", lambda p: Path(p))

    tr, te = load_kaggle_data(data_dir)

    assert isinstance(tr, pd.DataFrame)
    assert isinstance(te, pd.DataFrame)
    assert tr.shape[0] == 1
    assert te.shape[0] == 1
