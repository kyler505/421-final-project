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


def test_contracts_validate_submission_rows_ok() -> None:
    from src.contracts import validate_submission_rows

    validate_submission_rows([0, 1], [1, 0])


def test_contracts_validate_submission_rows_bad_label() -> None:
    from src.contracts import validate_submission_rows

    with pytest.raises(ValueError):
        validate_submission_rows([0], [2])


def test_manifest_roundtrip(tmp_path) -> None:
    from src.manifest import RunManifest, load_run_manifest, save_run_manifest

    m = RunManifest(backend="baseline", artifact_kind="sklearn_pickle", train_path="x.csv")
    path = tmp_path / "m.json"
    save_run_manifest(m, path)
    loaded = load_run_manifest(path)
    assert loaded.backend == "baseline"
    assert loaded.train_path == "x.csv"


def test_load_training_manifest(tmp_path) -> None:
    from src.data import get_texts_labels, load_training_manifest

    csv_path = tmp_path / "shard.csv"
    csv_path.write_text(
        'row_id,text,label\n0,"hello",1\n1,"world",0\n',
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"version": 1, "entries": [{"path": "shard.csv", "label_source": "gold", "split": "t"}]}',
        encoding="utf-8",
    )
    df = load_training_manifest(manifest_path)
    texts, labels = get_texts_labels(df)
    assert len(texts) == 2
    assert labels == [1, 0]
    assert df["label_source"].tolist() == ["gold", "gold"]
    assert df["split"].tolist() == ["t", "t"]


def test_cv_baseline_stratified_runs() -> None:
    from src.eval_cv import cv_baseline_stratified

    texts = [
        "patient has pneumonia",
        "tolerated diet",
        "diabetes mellitus history",
        "walked hallway",
        "acute kidney injury",
        "watching television",
        "hypertension controlled",
        "denies chest pain",
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    folds, means = cv_baseline_stratified(texts, labels, n_splits=4, random_state=0)
    assert len(folds) == 4
    assert "f1" in means


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
