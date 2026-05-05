from __future__ import annotations

import csv

import numpy as np
import src.predict as predict_module
from src.config import LogisticConfig, SelfTrainingConfig, VectorizerConfig
from src.data import read_labeled_dataset
import src.model as model_module
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


def test_predict_applies_saved_vectorizer_normalization(tmp_path, monkeypatch):
    class FakeBundle:
        threshold = 0.5
        metadata = {"vectorizer_cfg": {"replace_numbers": True}}

    seen = {}

    def fake_load_bundle(_path):
        return FakeBundle()

    def fake_predict_component_proba(_bundle, texts, _component):
        seen["texts"] = texts
        return [0.6]

    monkeypatch.setattr(predict_module, "load_bundle", fake_load_bundle)
    monkeypatch.setattr(predict_module, "predict_component_proba", fake_predict_component_proba)

    test_path = tmp_path / "test.csv"
    pred_path = tmp_path / "pred.csv"
    _write_csv(test_path, [{"row_id": "10", "text": "BP 120/80"}], include_label=False)

    predict_main([
        "--model",
        str(tmp_path / "model.joblib"),
        "--input",
        str(test_path),
        "--output",
        str(pred_path),
    ])

    assert seen["texts"] == ["bp <NUM>"]


def test_fit_full_pipeline_fixed_threshold_and_teacher_mode(monkeypatch):
    gold_texts = ["alpha", "beta", "gamma", "delta"]
    gold_labels = [1, 0, 1, 0]

    monkeypatch.setattr(
        model_module,
        "cross_validated_component_probs",
        lambda *args, **kwargs: {
            "baseline": np.array([0.9, 0.1, 0.8, 0.2], dtype=float),
            "ssl": np.array([0.85, 0.2, 0.75, 0.3], dtype=float),
            "embedding": np.array([0.88, 0.15, 0.82, 0.25], dtype=float),
        },
    )

    class FakeModel:
        def __init__(self, name):
            self.name = name
            self.pseudo_rows = 0

        def predict_proba(self, texts):
            base = 0.6 if self.name != "baseline" else 0.4
            return np.full(len(texts), base, dtype=float)

    monkeypatch.setattr(model_module, "fit_tfidf_model", lambda *args, **kwargs: FakeModel("tfidf"))
    monkeypatch.setattr(
        model_module,
        "fit_self_training_tfidf",
        lambda *args, **kwargs: (FakeModel("ssl"), model_module.SelfTrainingSummary(pseudo_rows=2)),
    )
    monkeypatch.setattr(model_module, "fit_embedding_model", lambda *args, **kwargs: FakeModel("embedding"))

    bundle, report = fit_full_pipeline(
        gold_texts,
        gold_labels,
        unlabeled_texts=["epsilon"],
        vectorizer_cfg=VectorizerConfig(max_features_word=100, max_features_char=100),
        ssl_cfg=SelfTrainingConfig(enabled=True, rank_mode=True),
        embedding_cfg=model_module.EmbeddingConfig(model_name="fake-model"),
        teacher_mode="tfidf",
        logistic_cfg=LogisticConfig(c=1.0),
        fixed_weights={"baseline": 0.5, "ssl": 0.3, "embedding": 0.2},
        fixed_threshold=0.43,
    )

    assert report["threshold"] == 0.43
    assert report["teacher_mode"] == "tfidf"
    assert bundle.threshold == 0.43
    assert bundle.weights == {"baseline": 0.5, "ssl": 0.3, "embedding": 0.2}
    assert bundle.embedding is not None
