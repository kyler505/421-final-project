from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from .config import EmbeddingConfig, LogisticConfig, MAX_WORDS, SelfTrainingConfig, VectorizerConfig
from .data import read_examples, read_labeled_dataset
from .model import fit_full_pipeline, predict_component_proba, save_bundle
from .text import normalize_for_vectorizer, normalize_text, truncate_words


def build_parser():
    p = argparse.ArgumentParser(description='Train ICD-codable text classifier')
    p.add_argument('--train', required=True, help='Path to labeled train CSV')
    p.add_argument('--unlabeled', nargs='*', default=[], help='Optional unlabeled CSV files with row_id,text')
    p.add_argument('--output', required=True, help='Path to write model bundle (.joblib)')
    p.add_argument('--summary-output', default=None, help='Optional JSON file for training summary')
    p.add_argument('--embedding-model', default=EmbeddingConfig().model_name, help='Local/offline embedding model name or path')
    p.add_argument('--no-self-training', action='store_true', help='Disable pseudo-labeling')
    p.add_argument('--no-embeddings', action='store_true', help='Disable embedding model')
    p.add_argument('--weight-step', type=float, default=0.1, help='Grid step for ensemble weights')
    p.add_argument('--threshold-step', type=float, default=0.01, help='Grid step for threshold search')
    p.add_argument('--max-unlabeled', type=int, default=0, help='Optional cap on unlabeled rows (0 = no cap)')
    p.add_argument('--replace-numbers', action='store_true', help='Replace numeric spans with <NUM> before vectorizing')
    p.add_argument('--logistic-c', type=float, default=1.0, help='Inverse regularization strength for logistic models')
    p.add_argument('--solver', default='liblinear', choices=['liblinear', 'lbfgs', 'saga'], help='LogisticRegression solver')
    p.add_argument('--penalty', default='l2', choices=['l1', 'l2', 'elasticnet'], help='LogisticRegression penalty')
    p.add_argument('--l1-ratio', type=float, default=None, help='Elastic-net l1_ratio when penalty=elasticnet')
    p.add_argument('--class-weight', default='balanced', choices=['balanced', 'none'], help='LogisticRegression class weighting')
    p.add_argument('--ssl-rounds', type=int, default=1, help='Number of SSL self-training rounds')
    p.add_argument('--ssl-positive-confidence', type=float, default=0.95, help='Positive pseudo-label confidence threshold')
    p.add_argument('--ssl-negative-confidence', type=float, default=0.05, help='Negative pseudo-label confidence threshold')
    p.add_argument('--ssl-gold-weight', type=float, default=5.0, help='Sample weight for gold labeled rows during SSL training')
    p.add_argument('--ssl-pseudo-weight', type=float, default=0.2, help='Sample weight for pseudo-labeled rows')
    p.add_argument('--ssl-max-pseudo-per-class-per-round', type=int, default=1000, help='Cap accepted pseudo-labels per class per round')
    p.add_argument('--ssl-rank-mode', action='store_true', help='Use rank-based pseudo-labeling when embeddings are enabled')
    p.add_argument('--ssl-rank-top-k', type=int, default=20, help='Top-K for rank-based pseudo-labeling')
    p.add_argument('--ssl-max-pool', type=int, default=500, help='Max unlabeled pool for rank-based pseudo-labeling')
    p.add_argument('--ssl-teacher-mode', choices=['tfidf', 'embedding'], default=None, help='Teacher used for rank-based SSL pseudo-labeling')
    p.add_argument('--ssl-teacher-embedding-model', default=None, help='Optional frozen embedding teacher for pseudo-label generation')
    p.add_argument('--fixed-plan-weights', action='store_true', help='Use the plan weights: baseline=.5 ssl=.3 embedding=.2')
    p.add_argument('--fixed-ensemble-threshold', type=float, default=None, help='Use a fixed ensemble threshold instead of searching')
    p.add_argument('--pseudo-manifest-output', default=None, help='Optional CSV file of pseudo-labeled rows')
    return p


def _load_unlabeled(paths: list[str], max_rows: int = 0, replace_numbers: bool = False) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_examples(path):
            text = truncate_words(normalize_for_vectorizer(row.text, replace_numbers=replace_numbers), MAX_WORDS)
            norm = normalize_text(text)
            if norm in seen:
                continue
            seen.add(norm)
            rows.append((row.row_id, text))
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def _print_component_metrics(bundle, texts, labels):
    for component in bundle.component_names():
        probs = predict_component_proba(bundle, texts, component)
        preds = (probs >= bundle.threshold).astype(int)
        print(f'[{component}] threshold={bundle.threshold:.3f} f1={f1_score(labels, preds, zero_division=0):.4f} acc={accuracy_score(labels, preds):.4f} precision={precision_score(labels, preds, zero_division=0):.4f} recall={recall_score(labels, preds, zero_division=0):.4f}')


def main(argv=None):
    args = build_parser().parse_args(argv)
    labeled = read_labeled_dataset(args.train, max_words=MAX_WORDS, replace_numbers=args.replace_numbers)
    row_ids, texts, labels = zip(*labeled)
    unlabeled_texts = _load_unlabeled(
        args.unlabeled,
        max_rows=args.max_unlabeled,
        replace_numbers=args.replace_numbers,
    )

    ssl_cfg = SelfTrainingConfig(
        enabled=not args.no_self_training,
        rounds=args.ssl_rounds,
        positive_confidence=args.ssl_positive_confidence,
        negative_confidence=args.ssl_negative_confidence,
        gold_weight=args.ssl_gold_weight,
        pseudo_weight=args.ssl_pseudo_weight,
        max_pseudo_per_class_per_round=args.ssl_max_pseudo_per_class_per_round,
        rank_mode=args.ssl_rank_mode,
        rank_top_k=args.ssl_rank_top_k,
        max_pool=args.ssl_max_pool,
    )
    teacher_cfg = EmbeddingConfig(model_name=args.ssl_teacher_embedding_model) if args.ssl_teacher_embedding_model else None
    embedding_cfg = None if args.no_embeddings else EmbeddingConfig(model_name=args.embedding_model)
    logistic_cfg = LogisticConfig(
        c=args.logistic_c,
        solver=args.solver,
        penalty=args.penalty,
        l1_ratio=args.l1_ratio,
        class_weight=None if args.class_weight == 'none' else 'balanced',
    )
    fixed_weights = None
    if args.fixed_plan_weights:
        fixed_weights = {'baseline': 0.5, 'ssl': 0.3, 'embedding': 0.2}
        if embedding_cfg is None:
            fixed_weights = {'baseline': 0.625, 'ssl': 0.375}
    bundle, report = fit_full_pipeline(
        list(texts),
        list(labels),
        unlabeled_texts=unlabeled_texts if unlabeled_texts else None,
        vectorizer_cfg=VectorizerConfig(replace_numbers=args.replace_numbers),
        ssl_cfg=ssl_cfg,
        embedding_cfg=embedding_cfg,
        teacher_cfg=teacher_cfg,
        teacher_mode=args.ssl_teacher_mode,
        logistic_cfg=logistic_cfg,
        weight_step=args.weight_step,
        threshold_step=args.threshold_step,
        fixed_weights=fixed_weights,
        fixed_threshold=args.fixed_ensemble_threshold,
        pseudo_manifest_output=args.pseudo_manifest_output,
    )

    probs = predict_component_proba(bundle, list(texts), 'ensemble')
    preds = (probs >= bundle.threshold).astype(int)
    print(f"threshold={bundle.threshold:.3f} weights={bundle.weights}")
    print(f"ensemble_f1={f1_score(labels, preds, zero_division=0):.4f}")
    print(f"ensemble_accuracy={accuracy_score(labels, preds):.4f}")
    print(f"ensemble_precision={precision_score(labels, preds, zero_division=0):.4f}")
    print(f"ensemble_recall={recall_score(labels, preds, zero_division=0):.4f}")
    print(classification_report(labels, preds, digits=4, zero_division=0))
    print('--- component holdout scores ---')
    _print_component_metrics(bundle, list(texts), list(labels))
    print('--- training summary ---')
    print(json.dumps(report, indent=2, sort_keys=True))

    save_bundle(bundle, args.output)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
