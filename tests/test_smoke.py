"""Smoke tests for the CSCE 421 final project scaffold."""

from __future__ import annotations

import tempfile

import pytest


def test_config_imports() -> None:
    from src import config

    assert config is not None


def test_data_imports() -> None:
    from src import data

    assert data is not None


def test_utils_imports() -> None:
    from src import utils

    assert utils is not None


def test_models_imports() -> None:
    from src.models import baseline
    from src.models import transformer

    assert baseline is not None
    assert transformer is not None


def test_baseline_model_fit_predict() -> None:
    from src.models.baseline import BaselineModel

    texts = [
        "Patient has diabetes mellitus.",
        "Patient walked in the hallway.",
        "History of congestive heart failure.",
        "Patient tolerated lunch well.",
    ]
    labels = [1, 0, 1, 0]

    model = BaselineModel()
    model.fit(texts, labels)
    preds = model.predict(texts)
    assert len(preds) == 4


def test_baseline_model_save_load(tmp_path) -> None:
    from src.models.baseline import BaselineModel

    texts = ["Diagnosis of pneumonia.", "Patient resting comfortably."]
    labels = [1, 0]

    model = BaselineModel()
    model.fit(texts, labels)

    model_path = tmp_path / "model.pkl"
    model.save(model_path)
    loaded = BaselineModel.load(model_path)
    preds = loaded.predict(texts)
    assert len(preds) == 2


def test_data_loader_with_labels_and_row_id() -> None:
    from src.data import load_train_data

    csv_content = """row_id,text,label
0,\"Patient has diabetes.\",1
1,\"Patient walked.\",0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
        handle.write(csv_content)
        path = handle.name

    df = load_train_data(path)
    assert list(df.columns) == ["row_id", "text", "label"]
    assert df["row_id"].tolist() == [0, 1]
    assert len(df) == 2


def test_data_loader_without_labels_keeps_row_id() -> None:
    from src.data import load_test_data

    csv_content = """row_id,text
0,\"Patient has diabetes.\"
1,\"Patient walked.\"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
        handle.write(csv_content)
        path = handle.name

    df = load_test_data(path)
    assert list(df.columns) == ["row_id", "text"]
    assert df["row_id"].tolist() == [0, 1]
    assert len(df) == 2


def test_submission_predictions_format(tmp_path) -> None:
    from src.utils import save_submission_predictions

    output_path = tmp_path / "preds.csv"
    save_submission_predictions(row_ids=[0, 1], predictions=[1, 0], output_path=output_path)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["row_id,prediction", "0,1", "1,0"]


def test_debug_predictions_format(tmp_path) -> None:
    from src.utils import save_debug_predictions

    output_path = tmp_path / "preds-debug.csv"
    save_debug_predictions(
        row_ids=[0, 1],
        texts=["Test 1", "Test 2"],
        predictions=[1, 0],
        probs=[0.9, 0.1],
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "row_id,text,prediction,probability"
    assert len(lines) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
