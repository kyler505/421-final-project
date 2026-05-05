from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut

from .config import DEFAULT_THRESHOLD, RANDOM_STATE, EmbeddingConfig, LogisticConfig, SelfTrainingConfig, VectorizerConfig
from .embeddings import get_embedding_encoder
from .text import normalize_text


@dataclass
class TfidfBinaryModel:
    word_vectorizer: Any
    char_vectorizer: Any
    classifier: Any
    training_rows: int = 0
    pseudo_rows: int = 0

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = hstack([self.word_vectorizer.transform(texts), self.char_vectorizer.transform(texts)])
        return self.classifier.predict_proba(X)[:, 1]


@dataclass
class EmbeddingBinaryModel:
    classifier: Any
    model_name: str
    max_length: int
    batch_size: int
    local_files_only: bool = True
    training_rows: int = 0
    pseudo_rows: int = 0

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        batch_size = max(self.batch_size, 64)
        encoder = get_embedding_encoder(
            self.model_name,
            max_length=self.max_length,
            batch_size=batch_size,
            local_files_only=self.local_files_only,
        )
        X = encoder.encode(texts, batch_size=batch_size)
        return self.classifier.predict_proba(X)[:, 1]


@dataclass
class EnsembleBundle:
    baseline: TfidfBinaryModel
    ssl: TfidfBinaryModel
    embedding: EmbeddingBinaryModel | None
    threshold: float = DEFAULT_THRESHOLD
    weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'EnsembleBundle':
        return joblib.load(path)

    def component_names(self) -> list[str]:
        names = ['baseline', 'ssl']
        if self.embedding is not None:
            names.append('embedding')
        return names


@dataclass
class SelfTrainingSummary:
    rounds_completed: int = 0
    pseudo_rows: int = 0
    pseudo_positive: int = 0
    pseudo_negative: int = 0
    confidence_positive: float = 0.0
    confidence_negative: float = 0.0
    gold_weight: float = 0.0
    pseudo_weight: float = 0.0
    max_pseudo_per_class_per_round: int = 0
    round_stats: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class WeightSearchResult:
    weights: dict[str, float]
    threshold: float
    f1: float
    accuracy: float
    precision: float
    recall: float


def metrics_at_threshold(probs: np.ndarray, labels: list[int] | np.ndarray, threshold: float) -> dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=int)
    preds = (np.asarray(probs, dtype=float) >= threshold).astype(int)
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(labels_arr, preds)),
        'precision': float(precision_score(labels_arr, preds, zero_division=0)),
        'recall': float(recall_score(labels_arr, preds, zero_division=0)),
        'f1': float(f1_score(labels_arr, preds, zero_division=0)),
        'positives': int(preds.sum()),
        'n_rows': int(len(preds)),
    }


def _build_vectorizers(cfg: VectorizerConfig | None = None):
    cfg = cfg or VectorizerConfig()
    word = TfidfVectorizer(
        analyzer='word',
        ngram_range=cfg.word_ngram_range,
        lowercase=False,
        min_df=cfg.min_df,
        max_features=cfg.max_features_word,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=cfg.char_ngram_range,
        lowercase=False,
        min_df=cfg.min_df,
        max_features=cfg.max_features_char,
        sublinear_tf=True,
    )
    return word, char


def _stack_features(word_vec, char_vec, texts: list[str]):
    return hstack([word_vec.transform(texts), char_vec.transform(texts)])


def _coerce_unlabeled_records(unlabeled_examples: list[str] | list[tuple[str, str]] | None) -> tuple[list[str], list[str]]:
    row_ids: list[str] = []
    texts: list[str] = []
    if not unlabeled_examples:
        return row_ids, texts
    first = unlabeled_examples[0]
    if isinstance(first, tuple):
        for row_id, text in unlabeled_examples:  # type: ignore[misc]
            row_ids.append(str(row_id) if row_id is not None else '')
            texts.append(str(text))
        return row_ids, texts
    for text in unlabeled_examples:  # type: ignore[assignment]
        row_ids.append('')
        texts.append(str(text))
    return row_ids, texts


def _dedupe_unlabeled_records(row_ids: list[str], texts: list[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    deduped_row_ids: list[str] = []
    deduped_texts: list[str] = []
    for row_id, text in zip(row_ids, texts):
        norm = normalize_text(text)
        if norm in seen:
            continue
        seen.add(norm)
        deduped_row_ids.append(row_id)
        deduped_texts.append(text)
    return deduped_row_ids, deduped_texts


def _gold_sample_weights(count: int, weight: float) -> list[float]:
    return [float(weight)] * count


def _fit_logistic(X, labels, sample_weight=None, cfg: LogisticConfig | None = None):
    cfg = cfg or LogisticConfig()
    kwargs = {
        'max_iter': cfg.max_iter,
        'C': cfg.c,
        'class_weight': cfg.class_weight,
        'random_state': RANDOM_STATE,
        'solver': cfg.solver,
        'penalty': cfg.penalty,
    }
    if cfg.penalty == 'elasticnet':
        kwargs['l1_ratio'] = cfg.l1_ratio
    clf = LogisticRegression(**kwargs)
    if sample_weight is None:
        clf.fit(X, labels)
    else:
        clf.fit(X, labels, sample_weight=np.asarray(sample_weight, dtype=float))
    return clf


def fit_tfidf_model(
    texts: list[str],
    labels: list[int],
    cfg: VectorizerConfig | None = None,
    sample_weight=None,
    logistic_cfg: LogisticConfig | None = None,
) -> TfidfBinaryModel:
    cfg = cfg or VectorizerConfig()
    word_vec, char_vec = _build_vectorizers(cfg)
    X_word = word_vec.fit_transform(texts)
    X_char = char_vec.fit_transform(texts)
    X = hstack([X_word, X_char])
    clf = _fit_logistic(X, labels, sample_weight=sample_weight, cfg=logistic_cfg)
    return TfidfBinaryModel(word_vec, char_vec, clf, training_rows=len(texts), pseudo_rows=0)


def _rank_based_pseudo_labels(
    model: Any,
    unlabeled_texts: list[str] | list[tuple[str, str]],
    top_k: int,
) -> tuple[list[str], list[int], list[float]]:
    """Select top-K most confident positive and negative predictions."""
    _, candidate_texts = _coerce_unlabeled_records(unlabeled_texts)
    probs = model.predict_proba(candidate_texts)
    # Sort indices by probability descending (most positive first)
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    pseudo_texts: list[str] = []
    labels: list[int] = []
    weights: list[float] = []
    seen: set[str] = set()
    # Top K positive
    for idx in ranked[:top_k]:
        txt = candidate_texts[idx]
        if txt not in seen:
            pseudo_texts.append(txt)
            labels.append(1)
            weights.append(float(probs[idx]))
            seen.add(txt)
    # Bottom K negative (most negative first)
    for idx in reversed(ranked[-top_k:]):
        txt = candidate_texts[idx]
        if txt not in seen:
            pseudo_texts.append(txt)
            labels.append(0)
            weights.append(1.0 - float(probs[idx]))
            seen.add(txt)
    return pseudo_texts, labels, weights


def _select_threshold_pseudo_labels(
    probs: np.ndarray,
    row_ids: list[str],
    texts: list[str],
    positive_confidence: float,
    negative_confidence: float,
    max_pseudo_per_class_per_round: int,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    positive_candidates: list[tuple[str, str, float]] = []
    negative_candidates: list[tuple[str, str, float]] = []
    for row_id, text, p in zip(row_ids, texts, probs):
        score = float(p)
        if score >= positive_confidence:
            positive_candidates.append((row_id, text, score))
        elif score <= negative_confidence:
            negative_candidates.append((row_id, text, score))
    positive_candidates.sort(key=lambda item: item[2], reverse=True)
    negative_candidates.sort(key=lambda item: item[2])
    cap = max(0, int(max_pseudo_per_class_per_round))
    if cap > 0:
        positive_candidates = positive_candidates[:cap]
        negative_candidates = negative_candidates[:cap]
    return positive_candidates, negative_candidates


def _resolve_teacher_model(
    gold_texts: list[str],
    gold_labels: list[int],
    vectorizer_cfg: VectorizerConfig,
    logistic_cfg: LogisticConfig | None,
    teacher_mode: str | None,
    teacher_cfg: EmbeddingConfig | None,
) -> tuple[Any | None, str | None]:
    mode = (teacher_mode or ('embedding' if teacher_cfg is not None else None))
    if mode is None:
        return None, None
    mode = mode.lower()
    if mode == 'tfidf':
        return fit_tfidf_model(gold_texts, gold_labels, vectorizer_cfg, logistic_cfg=logistic_cfg), 'tfidf'
    if mode == 'embedding':
        cfg = teacher_cfg or EmbeddingConfig()
        return fit_embedding_model(gold_texts, gold_labels, cfg), 'embedding'
    raise ValueError(f'Unknown teacher mode: {teacher_mode}')


def _similarity_based_pseudo_labels(
    gold_texts: list[str],
    gold_labels: list[int],
    unlabeled_texts: list[str] | list[tuple[str, str]],
    top_k: int,
    embedding_cfg: EmbeddingConfig,
    max_pool: int = 500,
) -> tuple[list[str], list[int], list[float]]:
    """Select pseudo-labels by embedding cosine similarity to gold examples.

    Finds unlabeled sentences most similar to positive gold examples (-> pos)
    and most similar to negative gold examples (-> neg).
    Avoids the LR classifier bottleneck — uses raw embedding similarity.
    """
    import random
    from sklearn.metrics.pairwise import cosine_similarity

    _, unlabeled_texts = _coerce_unlabeled_records(unlabeled_texts)
    # Limit unlabeled pool for speed
    pool = unlabeled_texts if len(unlabeled_texts) <= max_pool else random.sample(unlabeled_texts, max_pool)

    encoder = get_embedding_encoder(
        embedding_cfg.model_name,
        max_length=embedding_cfg.max_length,
        batch_size=max(embedding_cfg.batch_size, 64),
        local_files_only=embedding_cfg.local_files_only,
    )

    gold_emb = encoder.encode(gold_texts, batch_size=64)
    pool_emb = encoder.encode(pool, batch_size=64)

    # Separate gold embeddings by label
    pos_mask = np.array(gold_labels) == 1
    neg_mask = np.array(gold_labels) == 0
    gold_pos = gold_emb[pos_mask]
    gold_neg = gold_emb[neg_mask]

    # For each pooled sentence: max similarity to positive vs negative gold
    sim_pos = cosine_similarity(pool_emb, gold_pos).max(axis=1)
    sim_neg = cosine_similarity(pool_emb, gold_neg).max(axis=1)
    # Score: positive similarity minus negative similarity
    scores = sim_pos - sim_neg

    # Sort by score descending (most "positive-like" first, most "negative-like" last)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    texts: list[str] = []
    labels: list[int] = []
    weights: list[float] = []
    seen: set[str] = set()

    for idx in ranked[:top_k]:
        txt = pool[idx]
        if txt not in seen:
            texts.append(txt)
            labels.append(1)
            weights.append(max(0.0, min(1.0, float(scores[idx]))))
            seen.add(txt)

    for idx in reversed(ranked[-top_k:]):
        txt = pool[idx]
        if txt not in seen:
            texts.append(txt)
            labels.append(0)
            weights.append(max(0.0, min(1.0, float(-scores[idx]))))
            seen.add(txt)

    return texts, labels, weights


def fit_self_training_tfidf(
    gold_texts: list[str],
    gold_labels: list[int],
    unlabeled_texts: list[str] | list[tuple[str, str]] | None = None,
    vectorizer_cfg: VectorizerConfig | None = None,
    ssl_cfg: SelfTrainingConfig | None = None,
    logistic_cfg: LogisticConfig | None = None,
    teacher_mode: str | None = None,
    teacher_cfg: EmbeddingConfig | None = None,
    pseudo_manifest: list[dict[str, Any]] | None = None,
    pseudo_manifest_input: list[dict[str, Any]] | None = None,
) -> tuple[TfidfBinaryModel, SelfTrainingSummary]:
    vectorizer_cfg = vectorizer_cfg or VectorizerConfig()
    ssl_cfg = ssl_cfg or SelfTrainingConfig()
    summary = SelfTrainingSummary(
        confidence_positive=ssl_cfg.positive_confidence,
        confidence_negative=ssl_cfg.negative_confidence,
        gold_weight=ssl_cfg.gold_weight,
        pseudo_weight=ssl_cfg.pseudo_weight,
        max_pseudo_per_class_per_round=ssl_cfg.max_pseudo_per_class_per_round,
    )

    combined_texts = list(gold_texts)
    combined_labels = list(gold_labels)
    sample_weight = _gold_sample_weights(len(gold_labels), ssl_cfg.gold_weight)
    if pseudo_manifest_input:
        pseudo_texts, pseudo_labels, pseudo_weights = _manifest_to_pseudo_examples(pseudo_manifest_input)
        if pseudo_texts:
            combined_texts.extend(pseudo_texts)
            combined_labels.extend(pseudo_labels)
            sample_weight.extend([ssl_cfg.pseudo_weight * float(w) for w in pseudo_weights])
            summary.rounds_completed = 0
            summary.pseudo_rows = len(pseudo_texts)
            summary.pseudo_positive = int(sum(pseudo_labels))
            summary.pseudo_negative = int(len(pseudo_labels) - sum(pseudo_labels))
            summary.round_stats.append(
                {
                    'round': 0,
                    'teacher': 'manifest',
                    'candidate_positive': len([label for label in pseudo_labels if label == 1]),
                    'candidate_negative': len([label for label in pseudo_labels if label == 0]),
                    'accepted_positive': len([label for label in pseudo_labels if label == 1]),
                    'accepted_negative': len([label for label in pseudo_labels if label == 0]),
                    'accepted_total': len(pseudo_texts),
                    'remaining_after_round': 0,
                    'gold_weight': ssl_cfg.gold_weight,
                    'pseudo_weight': ssl_cfg.pseudo_weight,
                    'max_pseudo_per_class_per_round': ssl_cfg.max_pseudo_per_class_per_round,
                }
            )
            model = fit_tfidf_model(combined_texts, combined_labels, vectorizer_cfg, sample_weight=sample_weight, logistic_cfg=logistic_cfg)
            model.pseudo_rows = len(pseudo_texts)
            return model, summary
    remaining_row_ids, remaining_texts = _coerce_unlabeled_records(unlabeled_texts)
    remaining_row_ids, remaining_texts = _dedupe_unlabeled_records(remaining_row_ids, remaining_texts)
    if not ssl_cfg.enabled or not remaining_texts:
        model = fit_tfidf_model(combined_texts, combined_labels, vectorizer_cfg, sample_weight=sample_weight, logistic_cfg=logistic_cfg)
        model.pseudo_rows = 0
        return model, summary

    if ssl_cfg.rank_mode:
        teacher_model, _ = _resolve_teacher_model(
            gold_texts,
            gold_labels,
            vectorizer_cfg,
            logistic_cfg,
            teacher_mode,
            teacher_cfg,
        )
        if teacher_model is None:
            teacher_model = fit_tfidf_model(gold_texts, gold_labels, vectorizer_cfg, sample_weight=sample_weight, logistic_cfg=logistic_cfg)
        pseudo_texts, pseudo_labels, pseudo_weights = _rank_based_pseudo_labels(
            teacher_model,
            remaining_texts,
            top_k=ssl_cfg.rank_top_k,
        )
        if not pseudo_texts:
            model = fit_tfidf_model(combined_texts, combined_labels, vectorizer_cfg, sample_weight=sample_weight, logistic_cfg=logistic_cfg)
            model.pseudo_rows = 0
            return model, summary
        combined_texts.extend(pseudo_texts)
        combined_labels.extend(pseudo_labels)
        sample_weight.extend([ssl_cfg.pseudo_weight * float(w) for w in pseudo_weights])
        summary.rounds_completed = 1
        summary.pseudo_rows = len(pseudo_texts)
        summary.pseudo_positive = int(sum(pseudo_labels))
        summary.pseudo_negative = int(len(pseudo_labels) - sum(pseudo_labels))
        summary.round_stats.append(
            {
                'round': 1,
                'teacher': 'rank',
                'candidate_positive': len([label for label in pseudo_labels if label == 1]),
                'candidate_negative': len([label for label in pseudo_labels if label == 0]),
                'accepted_positive': len([label for label in pseudo_labels if label == 1]),
                'accepted_negative': len([label for label in pseudo_labels if label == 0]),
                'accepted_total': len(pseudo_texts),
                'remaining_after_round': 0,
                'gold_weight': ssl_cfg.gold_weight,
                'pseudo_weight': ssl_cfg.pseudo_weight,
                'max_pseudo_per_class_per_round': ssl_cfg.max_pseudo_per_class_per_round,
            }
        )
        final_model = fit_tfidf_model(combined_texts, combined_labels, vectorizer_cfg, sample_weight=sample_weight, logistic_cfg=logistic_cfg)
        final_model.pseudo_rows = summary.pseudo_rows
        return final_model, summary

    teacher_model: Any | None = None
    resolved_teacher_mode: str | None = None
    if teacher_mode is not None or teacher_cfg is not None:
        teacher_model, resolved_teacher_mode = _resolve_teacher_model(
            gold_texts,
            gold_labels,
            vectorizer_cfg,
            logistic_cfg,
            teacher_mode,
            teacher_cfg,
        )

    for round_idx in range(ssl_cfg.rounds):
        if not remaining_texts:
            summary.rounds_completed = round_idx
            break
        if teacher_model is None:
            scorer = fit_tfidf_model(
                combined_texts,
                combined_labels,
                vectorizer_cfg,
                sample_weight=sample_weight,
                logistic_cfg=logistic_cfg,
            )
            scorer_name = 'student'
        else:
            scorer = teacher_model
            scorer_name = resolved_teacher_mode or 'teacher'
        probs = scorer.predict_proba(remaining_texts)
        positive_candidates, negative_candidates = _select_threshold_pseudo_labels(
            probs,
            remaining_row_ids,
            remaining_texts,
            ssl_cfg.positive_confidence,
            ssl_cfg.negative_confidence,
            ssl_cfg.max_pseudo_per_class_per_round,
        )
        keep_texts: list[str] = []
        keep_labels: list[int] = []
        keep_weights: list[float] = []
        keep_set: set[str] = set()
        for row_id, text, p in positive_candidates:
            keep_texts.append(text)
            keep_labels.append(1)
            keep_weights.append(ssl_cfg.pseudo_weight)
            keep_set.add(normalize_text(text))
            if pseudo_manifest is not None:
                pseudo_manifest.append(
                    {
                        'round': round_idx + 1,
                        'source': scorer_name,
                        'row_id': row_id,
                        'text': text,
                        'label': 1,
                        'score': float(p),
                        'weight': float(ssl_cfg.pseudo_weight),
                    }
                )
        for row_id, text, p in negative_candidates:
            keep_texts.append(text)
            keep_labels.append(0)
            keep_weights.append(ssl_cfg.pseudo_weight)
            keep_set.add(normalize_text(text))
            if pseudo_manifest is not None:
                pseudo_manifest.append(
                    {
                        'round': round_idx + 1,
                        'source': scorer_name,
                        'row_id': row_id,
                        'text': text,
                        'label': 0,
                        'score': float(p),
                        'weight': float(ssl_cfg.pseudo_weight),
                    }
                )
        round_stat = {
            'round': round_idx + 1,
            'teacher': scorer_name,
            'candidate_positive': len(positive_candidates),
            'candidate_negative': len(negative_candidates),
            'accepted_positive': len(positive_candidates),
            'accepted_negative': len(negative_candidates),
            'accepted_total': len(keep_texts),
            'remaining_after_round': 0,
            'gold_weight': ssl_cfg.gold_weight,
            'pseudo_weight': ssl_cfg.pseudo_weight,
            'max_pseudo_per_class_per_round': ssl_cfg.max_pseudo_per_class_per_round,
        }
        if not keep_texts:
            summary.round_stats.append(
                {
                    **round_stat,
                    'accepted_positive': 0,
                    'accepted_negative': 0,
                    'accepted_total': 0,
                    'remaining_after_round': len(remaining_texts),
                }
            )
            summary.rounds_completed = round_idx
            break
        combined_texts.extend(keep_texts)
        combined_labels.extend(keep_labels)
        sample_weight.extend(keep_weights)
        summary.pseudo_rows += len(keep_texts)
        summary.pseudo_positive += int(sum(keep_labels))
        summary.pseudo_negative += int(len(keep_labels) - sum(keep_labels))
        remaining_pairs = [
            (row_id, text)
            for row_id, text in zip(remaining_row_ids, remaining_texts)
            if normalize_text(text) not in keep_set
        ]
        remaining_row_ids = [row_id for row_id, _ in remaining_pairs]
        remaining_texts = [text for _, text in remaining_pairs]
        summary.rounds_completed = round_idx + 1
        round_stat['remaining_after_round'] = len(remaining_texts)
        summary.round_stats.append(round_stat)
    final_model = fit_tfidf_model(combined_texts, combined_labels, vectorizer_cfg, sample_weight=sample_weight, logistic_cfg=logistic_cfg)
    final_model.pseudo_rows = summary.pseudo_rows
    return final_model, summary


def _write_pseudo_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['round', 'source', 'row_id', 'text', 'label', 'score', 'weight']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _read_pseudo_manifest(path: str | Path) -> list[dict[str, Any]]:
    in_path = Path(path)
    with in_path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _manifest_to_pseudo_examples(rows: list[dict[str, Any]]) -> tuple[list[str], list[int], list[float]]:
    texts: list[str] = []
    labels: list[int] = []
    weights: list[float] = []
    seen: set[str] = set()
    for row in rows:
        text = str(row.get('text', ''))
        norm = normalize_text(text)
        if not text or norm in seen:
            continue
        seen.add(norm)
        texts.append(text)
        labels.append(int(float(row.get('label', 0))))
        weights.append(float(row.get('weight', 0.2)))
    return texts, labels, weights


def fit_embedding_model(
    texts: list[str],
    labels: list[int],
    cfg: EmbeddingConfig | None = None,
    sample_weight=None,
) -> EmbeddingBinaryModel:
    cfg = cfg or EmbeddingConfig()
    encoder = get_embedding_encoder(
        cfg.model_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        local_files_only=cfg.local_files_only,
    )
    X = encoder.encode(texts, batch_size=cfg.batch_size)
    clf = _fit_logistic(X, labels, sample_weight=sample_weight, cfg=LogisticConfig())
    return EmbeddingBinaryModel(
        classifier=clf,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        local_files_only=cfg.local_files_only,
        training_rows=len(texts),
        pseudo_rows=0,
    )


def predict_component_proba(bundle: EnsembleBundle, texts: list[str], component: str) -> np.ndarray:
    component = component.lower()
    if component == 'baseline':
        return bundle.baseline.predict_proba(texts)
    if component == 'ssl':
        return bundle.ssl.predict_proba(texts)
    if component == 'embedding':
        if bundle.embedding is None:
            raise ValueError('Embedding component is unavailable in this bundle')
        return bundle.embedding.predict_proba(texts)
    if component == 'ensemble':
        return predict_ensemble_proba(bundle, texts)
    raise ValueError(f'Unknown component: {component}')


def predict_ensemble_proba(bundle: EnsembleBundle, texts: list[str]) -> np.ndarray:
    component_probs: list[np.ndarray] = []
    component_weights: list[float] = []
    for name in bundle.component_names():
        weight = float(bundle.weights.get(name, 0.0))
        if weight <= 0:
            continue
        component_probs.append(predict_component_proba(bundle, texts, name))
        component_weights.append(weight)
    if not component_probs:
        raise ValueError('No active ensemble weights found')
    total_weight = float(sum(component_weights))
    if total_weight <= 0:
        raise ValueError('Ensemble weights must sum to a positive value')
    blended = np.zeros(len(texts), dtype=float)
    for probs, weight in zip(component_probs, component_weights):
        blended += probs * (weight / total_weight)
    return blended


def _weights_grid(component_names: list[str], step: float = 0.1):
    if len(component_names) == 1:
        yield {component_names[0]: 1.0}
        return
    grid = [round(x, 10) for x in np.arange(0.0, 1.0 + 1e-9, step)]
    if len(component_names) == 2:
        a, b = component_names
        for wa in grid:
            wb = 1.0 - wa
            if wb < -1e-9:
                continue
            yield {a: float(wa), b: float(round(max(0.0, wb), 10))}
        return
    if len(component_names) == 3:
        a, b, c = component_names
        for wa in grid:
            for wb in grid:
                wc = 1.0 - wa - wb
                if wc < -1e-9:
                    continue
                yield {
                    a: float(wa),
                    b: float(wb),
                    c: float(round(max(0.0, wc), 10)),
                }
        return
    raise ValueError(f'Unsupported number of components for grid search: {component_names}')


def _score_thresholds(probs: np.ndarray, labels: np.ndarray, threshold_step: float = 0.01):
    best = None
    thresholds = np.arange(0.05, 0.95 + 1e-9, threshold_step)
    for threshold in thresholds:
        metrics = metrics_at_threshold(probs, labels, float(threshold))
        f1 = metrics['f1']
        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        candidate = WeightSearchResult(weights={}, threshold=float(threshold), f1=float(f1), accuracy=float(acc), precision=float(prec), recall=float(rec))
        if best is None:
            best = candidate
            continue
        if (candidate.f1, candidate.precision, candidate.accuracy, -candidate.threshold) > (
            best.f1,
            best.precision,
            best.accuracy,
            -best.threshold,
        ):
            best = candidate
    return best


def search_weights_and_threshold(
    component_probs: dict[str, np.ndarray],
    labels: list[int],
    step: float = 0.1,
    threshold_step: float = 0.01,
) -> WeightSearchResult:
    labels_arr = np.asarray(labels, dtype=int)
    component_names = list(component_probs)
    if not component_names:
        raise ValueError('No component probabilities provided')
    best: WeightSearchResult | None = None
    for weights in _weights_grid(component_names, step=step):
        total = sum(weights.values())
        if total <= 0:
            continue
        probs = np.zeros(len(labels_arr), dtype=float)
        for name, weight in weights.items():
            probs += component_probs[name] * (weight / total)
        candidate = _score_thresholds(probs, labels_arr, threshold_step=threshold_step)
        candidate = WeightSearchResult(
            weights=dict(weights),
            threshold=candidate.threshold,
            f1=candidate.f1,
            accuracy=candidate.accuracy,
            precision=candidate.precision,
            recall=candidate.recall,
        )
        if best is None:
            best = candidate
            continue
        if (candidate.f1, candidate.precision, candidate.accuracy, -candidate.threshold) > (
            best.f1,
            best.precision,
            best.accuracy,
            -best.threshold,
        ):
            best = candidate
    if best is None:
        raise RuntimeError('Unable to determine ensemble weights')
    return best


def cross_validated_component_probs(
    gold_texts: list[str],
    gold_labels: list[int],
    unlabeled_texts: list[str] | list[tuple[str, str]] | None,
    vectorizer_cfg: VectorizerConfig | None = None,
    ssl_cfg: SelfTrainingConfig | None = None,
    embedding_cfg: EmbeddingConfig | None = None,
    teacher_mode: str | None = None,
    teacher_cfg: EmbeddingConfig | None = None,
    logistic_cfg: LogisticConfig | None = None,
    pseudo_manifest_rows: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    vectorizer_cfg = vectorizer_cfg or VectorizerConfig()
    ssl_cfg = ssl_cfg or SelfTrainingConfig()
    loo = LeaveOneOut()
    labels_arr = np.asarray(gold_labels, dtype=int)
    gold_sample_weight = _gold_sample_weights(len(gold_texts), ssl_cfg.gold_weight)
    probs = {
        'baseline': np.zeros(len(gold_texts), dtype=float),
        'ssl': np.zeros(len(gold_texts), dtype=float),
    }
    use_embedding = embedding_cfg is not None
    if use_embedding:
        probs['embedding'] = np.zeros(len(gold_texts), dtype=float)
    for train_idx, test_idx in loo.split(gold_texts):
        train_texts = [gold_texts[i] for i in train_idx]
        train_labels = [int(labels_arr[i]) for i in train_idx]
        train_gold_weights = [gold_sample_weight[i] for i in train_idx]
        test_text = [gold_texts[test_idx[0]]]
        baseline = fit_tfidf_model(
            train_texts,
            train_labels,
            vectorizer_cfg,
            sample_weight=train_gold_weights,
            logistic_cfg=logistic_cfg,
        )
        ssl_model, _ = fit_self_training_tfidf(
            train_texts,
            train_labels,
            unlabeled_texts,
            vectorizer_cfg,
            ssl_cfg,
            logistic_cfg=logistic_cfg,
            teacher_mode=teacher_mode,
            teacher_cfg=teacher_cfg,
            pseudo_manifest_input=pseudo_manifest_rows,
        )
        probs['baseline'][test_idx[0]] = float(baseline.predict_proba(test_text)[0])
        probs['ssl'][test_idx[0]] = float(ssl_model.predict_proba(test_text)[0])
        if use_embedding:
            embedding_model = fit_embedding_model(
                train_texts,
                train_labels,
                embedding_cfg,
                sample_weight=train_gold_weights,
            )
            probs['embedding'][test_idx[0]] = float(embedding_model.predict_proba(test_text)[0])
    return probs


def fit_full_pipeline(
    gold_texts: list[str],
    gold_labels: list[int],
    unlabeled_texts: list[str] | list[tuple[str, str]] | None = None,
    vectorizer_cfg: VectorizerConfig | None = None,
    ssl_cfg: SelfTrainingConfig | None = None,
    embedding_cfg: EmbeddingConfig | None = None,
    teacher_mode: str | None = None,
    teacher_cfg: EmbeddingConfig | None = None,
    logistic_cfg: LogisticConfig | None = None,
    weight_step: float = 0.1,
    threshold_step: float = 0.01,
    fixed_weights: dict[str, float] | None = None,
    fixed_threshold: float | None = None,
    pseudo_manifest_output: str | None = None,
    pseudo_manifest_input: str | None = None,
) -> tuple[EnsembleBundle, dict[str, Any]]:
    vectorizer_cfg = vectorizer_cfg or VectorizerConfig()
    ssl_cfg = ssl_cfg or SelfTrainingConfig()

    manifest_rows = _read_pseudo_manifest(pseudo_manifest_input) if pseudo_manifest_input else None
    component_probs = cross_validated_component_probs(
        gold_texts,
        gold_labels,
        unlabeled_texts,
        vectorizer_cfg=vectorizer_cfg,
        ssl_cfg=ssl_cfg,
        embedding_cfg=embedding_cfg,
        teacher_mode=teacher_mode,
        teacher_cfg=teacher_cfg,
        logistic_cfg=logistic_cfg,
        pseudo_manifest_rows=manifest_rows,
    )
    if fixed_weights and fixed_threshold is not None:
        labels_arr = np.asarray(gold_labels, dtype=int)
        blended = np.zeros(len(labels_arr), dtype=float)
        total_weight = sum(float(fixed_weights.get(name, 0.0)) for name in component_probs)
        if total_weight <= 0:
            raise ValueError('fixed_weights must include at least one active component')
        for name, probs in component_probs.items():
            blended += probs * (float(fixed_weights.get(name, 0.0)) / total_weight)
        metrics = metrics_at_threshold(blended, labels_arr, fixed_threshold)
        best = WeightSearchResult(
            weights={name: float(fixed_weights.get(name, 0.0)) for name in component_probs},
            threshold=float(fixed_threshold),
            f1=float(metrics['f1']),
            accuracy=float(metrics['accuracy']),
            precision=float(metrics['precision']),
            recall=float(metrics['recall']),
        )
    elif fixed_weights:
        labels_arr = np.asarray(gold_labels, dtype=int)
        blended = np.zeros(len(labels_arr), dtype=float)
        total_weight = sum(float(fixed_weights.get(name, 0.0)) for name in component_probs)
        if total_weight <= 0:
            raise ValueError('fixed_weights must include at least one active component')
        for name, probs in component_probs.items():
            blended += probs * (float(fixed_weights.get(name, 0.0)) / total_weight)
        threshold_result = _score_thresholds(blended, labels_arr, threshold_step=threshold_step)
        best = WeightSearchResult(
            weights={name: float(fixed_weights.get(name, 0.0)) for name in component_probs},
            threshold=threshold_result.threshold,
            f1=threshold_result.f1,
            accuracy=threshold_result.accuracy,
            precision=threshold_result.precision,
            recall=threshold_result.recall,
        )
    else:
        best = search_weights_and_threshold(component_probs, gold_labels, step=weight_step, threshold_step=threshold_step)

    labels_arr = np.asarray(gold_labels, dtype=int)
    evaluation_threshold = best.threshold if fixed_threshold is not None else None
    component_cv_metrics = {}
    for name, probs in component_probs.items():
        if evaluation_threshold is None:
            component_threshold = _score_thresholds(probs, labels_arr, threshold_step).threshold
        else:
            component_threshold = evaluation_threshold
        component_cv_metrics[name] = metrics_at_threshold(probs, gold_labels, component_threshold)

    gold_sample_weight = _gold_sample_weights(len(gold_labels), ssl_cfg.gold_weight)
    baseline = fit_tfidf_model(gold_texts, gold_labels, vectorizer_cfg, sample_weight=gold_sample_weight, logistic_cfg=logistic_cfg)

    # Similarity-based SSL: use embedding similarity to find pseudo-labels
    # when absolute thresholds can't produce any (tiny dataset problem)
    ssl_summary = SelfTrainingSummary(
        confidence_positive=ssl_cfg.positive_confidence,
        confidence_negative=ssl_cfg.negative_confidence,
    )
    pseudo_manifest: list[dict[str, Any]] = []
    teacher_model, resolved_teacher_mode = _resolve_teacher_model(
        gold_texts,
        gold_labels,
        vectorizer_cfg,
        logistic_cfg,
        teacher_mode,
        teacher_cfg,
    )
    if ssl_cfg.enabled and ssl_cfg.rank_mode and unlabeled_texts and teacher_model is not None:
        pseudo_texts, pseudo_labels, pseudo_weights = _rank_based_pseudo_labels(
            teacher_model,
            unlabeled_texts,
            top_k=ssl_cfg.rank_top_k,
        )
        if pseudo_texts:
            ssl_summary.rounds_completed = 1
            ssl_summary.pseudo_rows = len(pseudo_texts)
            ssl_summary.pseudo_positive = sum(1 for l in pseudo_labels if l == 1)
            ssl_summary.pseudo_negative = sum(1 for l in pseudo_labels if l == 0)
            combined_texts = list(gold_texts) + pseudo_texts
            combined_labels = list(gold_labels) + pseudo_labels
            combined_weights = _gold_sample_weights(len(gold_labels), ssl_cfg.gold_weight) + [ssl_cfg.pseudo_weight * float(w) for w in pseudo_weights]
            ssl_model = fit_tfidf_model(
                combined_texts,
                combined_labels,
                vectorizer_cfg,
                sample_weight=combined_weights,
                logistic_cfg=logistic_cfg,
            )
            ssl_model.pseudo_rows = len(pseudo_texts)
        else:
            ssl_model, ssl_summary2 = fit_self_training_tfidf(
                gold_texts,
                gold_labels,
                unlabeled_texts,
                vectorizer_cfg,
                ssl_cfg,
                logistic_cfg=logistic_cfg,
                teacher_mode=teacher_mode,
                teacher_cfg=teacher_cfg,
                pseudo_manifest=pseudo_manifest if pseudo_manifest_output else None,
            )
            ssl_summary = ssl_summary2
    elif ssl_cfg.enabled and ssl_cfg.rank_mode and unlabeled_texts and embedding_cfg is not None:
        pseudo_texts, pseudo_labels, pseudo_weights = _similarity_based_pseudo_labels(
            gold_texts,
            gold_labels,
            unlabeled_texts,
            top_k=ssl_cfg.rank_top_k,
            embedding_cfg=embedding_cfg,
            max_pool=ssl_cfg.max_pool,
        )
        if pseudo_texts:
            ssl_summary.rounds_completed = 1
            ssl_summary.pseudo_rows = len(pseudo_texts)
            ssl_summary.pseudo_positive = sum(1 for l in pseudo_labels if l == 1)
            ssl_summary.pseudo_negative = sum(1 for l in pseudo_labels if l == 0)
            # Train SSL-TF-IDF on gold + pseudo-labels
            combined_texts = list(gold_texts) + pseudo_texts
            combined_labels = list(gold_labels) + pseudo_labels
            combined_weights = _gold_sample_weights(len(gold_labels), ssl_cfg.gold_weight) + [ssl_cfg.pseudo_weight] * len(pseudo_labels)
            ssl_model = fit_tfidf_model(
                combined_texts,
                combined_labels,
                vectorizer_cfg,
                sample_weight=combined_weights,
                logistic_cfg=logistic_cfg,
            )
            ssl_model.pseudo_rows = len(pseudo_texts)
        else:
            ssl_model, ssl_summary2 = fit_self_training_tfidf(
                gold_texts,
                gold_labels,
                unlabeled_texts,
                vectorizer_cfg,
                ssl_cfg,
                logistic_cfg=logistic_cfg,
                teacher_mode=teacher_mode,
                teacher_cfg=teacher_cfg,
                pseudo_manifest=pseudo_manifest if pseudo_manifest_output else None,
            )
            ssl_summary = ssl_summary2
    else:
        ssl_model, ssl_summary = fit_self_training_tfidf(
            gold_texts,
            gold_labels,
            unlabeled_texts,
            vectorizer_cfg,
            ssl_cfg,
            logistic_cfg=logistic_cfg,
            teacher_mode=teacher_mode,
            teacher_cfg=teacher_cfg,
            pseudo_manifest=pseudo_manifest if pseudo_manifest_output else None,
            pseudo_manifest_input=manifest_rows,
        )

    embedding_model = (
        fit_embedding_model(gold_texts, gold_labels, embedding_cfg, sample_weight=gold_sample_weight)
        if embedding_cfg is not None
        else None
    )
    if pseudo_manifest_output:
        _write_pseudo_manifest(pseudo_manifest_output, pseudo_manifest)

    bundle = EnsembleBundle(
        baseline=baseline,
        ssl=ssl_model,
        embedding=embedding_model,
        threshold=best.threshold,
        weights=best.weights,
        metadata={
            'vectorizer_cfg': asdict(vectorizer_cfg),
            'ssl_cfg': asdict(ssl_cfg),
            'logistic_cfg': asdict(logistic_cfg or LogisticConfig()),
            'embedding_cfg': asdict(embedding_cfg) if embedding_cfg is not None else None,
            'teacher_cfg': asdict(teacher_cfg) if teacher_cfg is not None and resolved_teacher_mode == 'embedding' else None,
            'teacher_mode': resolved_teacher_mode,
            'cv_metrics': {
                'f1': best.f1,
                'accuracy': best.accuracy,
                'precision': best.precision,
                'recall': best.recall,
            },
            'ssl_summary': asdict(ssl_summary),
            'pseudo_manifest_output': pseudo_manifest_output,
            'component_cv_metrics': component_cv_metrics,
            'component_cv_probabilities': {k: v.tolist() for k, v in component_probs.items()},
        },
    )
    report = {
        'threshold': best.threshold,
        'weights': best.weights,
        'cv_metrics': {
            'f1': best.f1,
            'accuracy': best.accuracy,
            'precision': best.precision,
            'recall': best.recall,
        },
        'ssl_summary': asdict(ssl_summary),
        'pseudo_manifest_output': pseudo_manifest_output,
        'component_names': bundle.component_names(),
        'component_cv_metrics': component_cv_metrics,
        'vectorizer_cfg': asdict(vectorizer_cfg),
        'logistic_cfg': asdict(logistic_cfg or LogisticConfig()),
        'embedding_cfg': asdict(embedding_cfg) if embedding_cfg is not None else None,
        'teacher_cfg': asdict(teacher_cfg) if teacher_cfg is not None and resolved_teacher_mode == 'embedding' else None,
        'teacher_mode': resolved_teacher_mode,
    }
    return bundle, report


def save_bundle(bundle: EnsembleBundle, path: str) -> None:
    bundle.save(path)


def load_bundle(path: str) -> EnsembleBundle:
    return EnsembleBundle.load(path)
