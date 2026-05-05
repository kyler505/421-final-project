from __future__ import annotations

import csv

from src.config import LogisticConfig, SelfTrainingConfig, VectorizerConfig
from src.data import read_labeled_dataset
from src.model import fit_full_pipeline, metrics_at_threshold, predict_component_proba
from src.predict import main as predict_main
from src.text import has_strong_negation, normalize_for_vectorizer, truncate_words


def _write_csv(path, rows, include_label=True):
    fields = ["row_id", "text"] + (["label"] if include_label else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_text_normalization_and_negation():
    assert normalize_for_vectorizer("BP 120/80", replace_numbers=True) == "bp <NUM>"
    assert truncate_words("one two three", max_words=2) == "one two"
    assert has_strong_negation("No evidence of pneumonia.")
    assert not has_strong_negation("History of pneumonia.")


def test_train_save_predict_debug(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    pred_path = tmp_path / "pred.csv"
    debug_path = tmp_path / "debug.csv"
    rows = [
        {"row_id": "0", "text": "pneumonia treated with antibiotics", "label": 1},
        {"row_id": "1", "text": "sepsis with organ failure", "label": 1},
        {"row_id": "2", "text": "patient resting comfortably", "label": 0},
        {"row_id": "3", "text": "no evidence of acute disease", "label": 0},
    ]
    _write_csv(train_path, rows)
    _write_csv(test_path, [{"row_id": "10", "text": "pneumonia treated"}, {"row_id": "11", "text": "resting comfortably"}], include_label=False)

    _, texts, labels = zip(*read_labeled_dataset(train_path))
    bundle, report = fit_full_pipeline(
        list(texts),
        list(labels),
        vectorizer_cfg=VectorizerConfig(max_features_word=100, max_features_char=100),
        ssl_cfg=SelfTrainingConfig(enabled=False),
        embedding_cfg=None,
        logistic_cfg=LogisticConfig(c=1.0),
        fixed_weights={"baseline": 1.0},
    )
    bundle.save(model_path)
    assert report["component_cv_metrics"]["baseline"]["n_rows"] == 4
    probs = predict_component_proba(bundle, list(texts), "baseline")
    assert metrics_at_threshold(probs, list(labels), bundle.threshold)["n_rows"] == 4

    predict_main([
        "--model",
        str(model_path),
        "--input",
        str(test_path),
        "--output",
        str(pred_path),
        "--debug-output",
        str(debug_path),
    ])
    with pred_path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["row_id", "prediction"]
    with debug_path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["row_id", "text", "probability", "prediction"]
