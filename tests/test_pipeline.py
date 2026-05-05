from __future__ import annotations

import csv

import numpy as np
import src.predict as predict_module
from src.config import LogisticConfig, SelfTrainingConfig, VectorizerConfig
from src.data import read_labeled_dataset
import src.model as model_module
from src.model import fit_full_pipeline, metrics_at_threshold, predict_component_proba
from src.predict import main as predict_main
from src.unlabeled_cache import UnlabeledCandidate, filter_candidates, read_cache, write_cache
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


def test_fit_full_pipeline_writes_pseudo_manifest(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self, probs):
            self._probs = np.asarray(probs, dtype=float)
            self.pseudo_rows = 0

        def predict_proba(self, texts):
            if len(self._probs) >= len(texts):
                return self._probs[: len(texts)]
            reps = int(np.ceil(len(texts) / len(self._probs)))
            return np.tile(self._probs, reps)[: len(texts)]

    teacher = FakeModel([0.97, 0.96, 0.04, 0.03])

    monkeypatch.setattr(
        model_module,
        "_resolve_teacher_model",
        lambda *args, **kwargs: (teacher, "tfidf"),
    )
    monkeypatch.setattr(model_module, "fit_tfidf_model", lambda *args, **kwargs: FakeModel([0.5, 0.5, 0.5, 0.5]))
    monkeypatch.setattr(model_module, "fit_embedding_model", lambda *args, **kwargs: FakeModel([0.5, 0.5, 0.5, 0.5]))
    monkeypatch.setattr(
        model_module,
        "cross_validated_component_probs",
        lambda *args, **kwargs: {
            "baseline": np.array([0.9, 0.1, 0.8, 0.2], dtype=float),
            "ssl": np.array([0.88, 0.12, 0.77, 0.25], dtype=float),
            "embedding": np.array([0.87, 0.14, 0.79, 0.24], dtype=float),
        },
    )

    manifest_path = tmp_path / "pseudo_manifest.csv"
    bundle, report = fit_full_pipeline(
        ["alpha positive", "beta negative", "gamma positive", "delta negative"],
        [1, 0, 1, 0],
        unlabeled_texts=[("u1", "one"), ("u2", "two"), ("u3", "three"), ("u4", "four")],
        vectorizer_cfg=VectorizerConfig(max_features_word=100, max_features_char=100),
        ssl_cfg=SelfTrainingConfig(enabled=True, rounds=1, rank_mode=False, positive_confidence=0.95, negative_confidence=0.05),
        embedding_cfg=None,
        teacher_mode="tfidf",
        logistic_cfg=LogisticConfig(c=1.0),
        fixed_weights={"baseline": 1.0},
        fixed_threshold=0.5,
        pseudo_manifest_output=str(manifest_path),
    )

    assert report["pseudo_manifest_output"] == str(manifest_path)
    assert bundle.metadata["teacher_mode"] == "tfidf"
    assert manifest_path.exists()
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 4
    assert rows[0]["source"] == "tfidf"
    assert rows[0]["row_id"] == "u1"


def test_fit_self_training_tfidf_stops_when_unlabeled_pool_is_empty(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self, probs):
            self._probs = np.asarray(probs, dtype=float)
            self.pseudo_rows = 0

        def predict_proba(self, texts):
            if not texts:
                return np.asarray([], dtype=float)
            reps = int(np.ceil(len(texts) / len(self._probs)))
            return np.tile(self._probs, reps)[: len(texts)]

    teacher = FakeModel([0.99, 0.98, 0.02, 0.01])

    monkeypatch.setattr(
        model_module,
        "_resolve_teacher_model",
        lambda *args, **kwargs: (teacher, "tfidf"),
    )
    monkeypatch.setattr(model_module, "fit_tfidf_model", lambda *args, **kwargs: FakeModel([0.5, 0.5, 0.5, 0.5]))

    manifest_path = tmp_path / "pseudo.csv"
    model, summary = model_module.fit_self_training_tfidf(
        ["alpha positive", "beta negative", "gamma positive", "delta negative"],
        [1, 0, 1, 0],
        unlabeled_texts=[("u1", "one"), ("u2", "two"), ("u3", "three"), ("u4", "four")],
        ssl_cfg=SelfTrainingConfig(enabled=True, rounds=2, positive_confidence=0.5, negative_confidence=0.5, pseudo_weight=0.2),
        teacher_mode="tfidf",
        pseudo_manifest=[],
    )

    assert model.pseudo_rows == 4
    assert summary.rounds_completed == 1


def test_fit_self_training_tfidf_respects_per_class_caps_and_gold_weight(monkeypatch):
    class FakeModel:
        def __init__(self, probs):
            self._probs = np.asarray(probs, dtype=float)
            self.pseudo_rows = 0

        def predict_proba(self, texts):
            if not texts:
                return np.asarray([], dtype=float)
            reps = int(np.ceil(len(texts) / len(self._probs)))
            return np.tile(self._probs, reps)[: len(texts)]

    teacher = FakeModel([0.99, 0.98, 0.97, 0.04, 0.03, 0.02])
    fit_calls = []

    def fake_fit_tfidf_model(texts, labels, cfg=None, sample_weight=None, logistic_cfg=None):
        fit_calls.append(list(sample_weight) if sample_weight is not None else None)
        return FakeModel([0.5] * max(1, len(texts)))

    monkeypatch.setattr(model_module, "_resolve_teacher_model", lambda *args, **kwargs: (teacher, "tfidf"))
    monkeypatch.setattr(model_module, "fit_tfidf_model", fake_fit_tfidf_model)

    model, summary = model_module.fit_self_training_tfidf(
        ["alpha positive", "beta negative", "gamma positive", "delta negative"],
        [1, 0, 1, 0],
        unlabeled_texts=[
            ("u1", "one"),
            ("u2", "two"),
            ("u3", "three"),
            ("u4", "four"),
            ("u5", "five"),
            ("u6", "six"),
        ],
        ssl_cfg=SelfTrainingConfig(
            enabled=True,
            rounds=1,
            positive_confidence=0.95,
            negative_confidence=0.05,
            gold_weight=7.0,
            pseudo_weight=0.2,
            max_pseudo_per_class_per_round=1,
        ),
        teacher_mode="tfidf",
        pseudo_manifest=[],
    )

    assert model.pseudo_rows == 2
    assert summary.pseudo_positive == 1
    assert summary.pseudo_negative == 1
    assert summary.round_stats[0]["accepted_positive"] == 1
    assert summary.round_stats[0]["accepted_negative"] == 1
    assert summary.round_stats[0]["gold_weight"] == 7.0
    assert any(weights == [7.0, 7.0, 7.0, 7.0, 0.2, 0.2] for weights in fit_calls if weights is not None)


def test_unlabeled_cache_filters_and_roundtrips(tmp_path):
    rows = [
        UnlabeledCandidate(row_id="1", note_id="n1", sentence="electronically signed by", sentence_hash="a", word_count=3),
        UnlabeledCandidate(row_id="2", note_id="n1", sentence="pneumonia treated with antibiotics", sentence_hash="b", word_count=4),
        UnlabeledCandidate(row_id="3", note_id="n1", sentence="pneumonia treated with antibiotics", sentence_hash="c", word_count=4),
    ]
    kept, dropped = filter_candidates(rows, per_note_cap=10, candidate_cap=10)
    assert len(kept) == 1
    assert dropped[0].drop_reason == "too_short"
    assert any(r.drop_reason == "duplicate" for r in dropped)

    cache_path = tmp_path / "unlabeled_sentences.parquet"
    write_cache(kept, cache_path)
    loaded = read_cache(cache_path if cache_path.exists() else cache_path.with_suffix(".csv.gz"))
    assert loaded[0].sentence == "pneumonia treated with antibiotics"


def test_fit_full_pipeline_uses_pseudo_manifest_input(monkeypatch):
    manifest_rows = [
        {"round": "1", "source": "manifest", "row_id": "u1", "text": "one", "label": "1", "score": "0.99", "weight": "0.2"},
        {"round": "1", "source": "manifest", "row_id": "u2", "text": "two", "label": "0", "score": "0.01", "weight": "0.2"},
    ]
    load_calls = []

    monkeypatch.setattr(model_module, "_read_pseudo_manifest", lambda path: load_calls.append(path) or manifest_rows)
    monkeypatch.setattr(
        model_module,
        "cross_validated_component_probs",
        lambda *args, **kwargs: {
            "baseline": np.array([0.9, 0.1], dtype=float),
            "ssl": np.array([0.85, 0.15], dtype=float),
        },
    )

    class FakeModel:
        def __init__(self):
            self.pseudo_rows = 0
        def predict_proba(self, texts):
            return np.full(len(texts), 0.5, dtype=float)

    monkeypatch.setattr(model_module, "fit_tfidf_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(model_module, "fit_embedding_model", lambda *args, **kwargs: None)

    bundle, report = fit_full_pipeline(
        ["alpha", "beta"],
        [1, 0],
        unlabeled_texts=None,
        vectorizer_cfg=VectorizerConfig(max_features_word=10, max_features_char=10),
        ssl_cfg=SelfTrainingConfig(enabled=True),
        embedding_cfg=None,
        logistic_cfg=LogisticConfig(c=1.0),
        fixed_weights={"baseline": 1.0, "ssl": 0.0},
        fixed_threshold=0.5,
        pseudo_manifest_input="manifest.csv",
    )

    assert load_calls == ["manifest.csv"]
    assert report["ssl_summary"]["pseudo_rows"] == 2
    assert bundle.ssl.pseudo_rows == 2
