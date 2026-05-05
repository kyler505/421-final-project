from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import EmbeddingConfig, LogisticConfig, MAX_WORDS, SelfTrainingConfig, VectorizerConfig
from .data import read_labeled_dataset, read_unlabeled_records
from .model import cross_validated_component_probs, fit_full_pipeline, metrics_at_threshold, save_bundle, search_weights_and_threshold
from .text import normalize_text


def build_parser():
    p = argparse.ArgumentParser(description='Sweep ICD-codable model candidates with LOOCV')
    p.add_argument('--train', required=True, help='Path to labeled training CSV')
    p.add_argument('--unlabeled', nargs='*', default=[], help='Optional unlabeled CSV files')
    p.add_argument('--output', required=True, help='JSON summary path')
    p.add_argument('--model-output', default=None, help='Optional path to save the best trained bundle')
    p.add_argument('--max-unlabeled', type=int, default=0, help='Optional cap on unlabeled rows')
    p.add_argument('--replace-numbers', action='store_true')
    p.add_argument('--embedding-model', default=EmbeddingConfig().model_name, help='Local/offline embedding model name or path')
    p.add_argument('--no-embeddings', action='store_true', help='Disable embedding model in the final ensemble')
    p.add_argument('--include-ssl', action='store_true', help='Include SSL component candidates')
    p.add_argument('--weight-step', type=float, default=0.1)
    p.add_argument('--threshold-step', type=float, default=0.01)
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
    p.add_argument('--pseudo-manifest-output', default=None, help='Optional CSV file of pseudo-labeled rows')
    p.add_argument('--pseudo-manifest-input', default=None, help='Optional CSV manifest of precomputed pseudo-labeled rows')
    p.add_argument('--calibration-mode', action='store_true', help='Use a narrower calibration-only grid')
    return p


def _load_unlabeled(paths: list[str], max_rows: int = 0, replace_numbers: bool = False) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        for row_id, text in read_unlabeled_records(path, max_words=MAX_WORDS, replace_numbers=replace_numbers):
            norm = normalize_text(text)
            if norm in seen:
                continue
            seen.add(norm)
            rows.append((row_id, text))
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def _candidate_grid(replace_numbers: bool, calibration_mode: bool = False):
    if calibration_mode:
        vectorizers = [
            VectorizerConfig((1, 3), (3, 5), 40000, 30000, 1, replace_numbers),
        ]
        c_values = [0.25, 1.0, 3.0]
    else:
        vectorizers = [
            VectorizerConfig((1, 2), (3, 5), 5000, 5000, 1, replace_numbers),
            VectorizerConfig((1, 3), (3, 5), 20000, 10000, 1, replace_numbers),
            VectorizerConfig((1, 2), (3, 4), 10000, 5000, 1, replace_numbers),
        ]
        c_values = [0.25, 1.0, 3.0, 10.0]
    logistics = [
        LogisticConfig(c=c, solver='liblinear', penalty='l2', class_weight=class_weight)
        for class_weight in (None, 'balanced')
        for c in c_values
    ]
    if not calibration_mode:
        logistics.append(LogisticConfig(c=3.0, solver='saga', penalty='elasticnet', l1_ratio=0.1, max_iter=8000, class_weight=None))
    for vi, vectorizer_cfg in enumerate(vectorizers):
        for li, logistic_cfg in enumerate(logistics):
            yield f'v{vi}_lr{li}', vectorizer_cfg, logistic_cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    labeled = read_labeled_dataset(args.train, max_words=MAX_WORDS, replace_numbers=args.replace_numbers)
    _, texts, labels = zip(*labeled)
    texts = list(texts)
    labels = list(labels)
    unlabeled = _load_unlabeled(args.unlabeled, args.max_unlabeled, replace_numbers=args.replace_numbers)
    ssl_cfg = SelfTrainingConfig(
        enabled=args.include_ssl,
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
    pseudo_manifest_rows = None
    if args.pseudo_manifest_input:
        from .model import _read_pseudo_manifest
        pseudo_manifest_rows = _read_pseudo_manifest(args.pseudo_manifest_input)

    results = []
    for name, vectorizer_cfg, logistic_cfg in _candidate_grid(args.replace_numbers, calibration_mode=args.calibration_mode):
        print(f"[candidate:start] {name} vectorizer={asdict(vectorizer_cfg)} logistic={asdict(logistic_cfg)}", flush=True)
        component_probs = cross_validated_component_probs(
            texts,
            labels,
            unlabeled if unlabeled else None,
            vectorizer_cfg=vectorizer_cfg,
            ssl_cfg=ssl_cfg,
            embedding_cfg=embedding_cfg,
            teacher_mode=args.ssl_teacher_mode,
            teacher_cfg=teacher_cfg,
            logistic_cfg=logistic_cfg,
            pseudo_manifest_rows=pseudo_manifest_rows,
        )
        best = search_weights_and_threshold(component_probs, labels, step=args.weight_step, threshold_step=args.threshold_step)
        component_metrics = {}
        for component, probs in component_probs.items():
            component_best = search_weights_and_threshold({component: probs}, labels, step=1.0, threshold_step=args.threshold_step)
            component_metrics[component] = metrics_at_threshold(probs, labels, component_best.threshold)
        results.append(
            {
                'name': name,
                'vectorizer_cfg': asdict(vectorizer_cfg),
                'logistic_cfg': asdict(logistic_cfg),
                'weights': best.weights,
                'threshold': best.threshold,
                'metrics': {
                    'accuracy': best.accuracy,
                    'precision': best.precision,
                    'recall': best.recall,
                    'f1': best.f1,
                },
                'component_metrics': component_metrics,
            }
        )
        print(f"[candidate:end] {name} metrics={results[-1]['metrics']} weights={results[-1]['weights']} threshold={results[-1]['threshold']:.3f}", flush=True)

    results.sort(key=lambda r: (r['metrics']['f1'], r['metrics']['precision'], r['metrics']['accuracy'], r['metrics']['recall']), reverse=True)
    summary = {'best': results[0], 'results': results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    if args.model_output:
        best = results[0]
        bundle, _ = fit_full_pipeline(
            texts,
            labels,
            unlabeled_texts=unlabeled if unlabeled else None,
            vectorizer_cfg=VectorizerConfig(**best['vectorizer_cfg']),
            ssl_cfg=ssl_cfg,
            embedding_cfg=embedding_cfg,
            teacher_mode=args.ssl_teacher_mode,
            teacher_cfg=teacher_cfg,
            logistic_cfg=LogisticConfig(**best['logistic_cfg']),
            fixed_weights=best['weights'],
            threshold_step=args.threshold_step,
            pseudo_manifest_output=args.pseudo_manifest_output,
            pseudo_manifest_input=args.pseudo_manifest_input,
        )
        save_bundle(bundle, args.model_output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
