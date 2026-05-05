from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import LogisticConfig, MAX_WORDS, SelfTrainingConfig, VectorizerConfig
from .data import read_examples, read_labeled_dataset
from .model import cross_validated_component_probs, fit_full_pipeline, metrics_at_threshold, save_bundle, search_weights_and_threshold
from .text import deduplicate_texts


def build_parser():
    p = argparse.ArgumentParser(description='Sweep ICD-codable model candidates with LOOCV')
    p.add_argument('--train', required=True, help='Path to labeled training CSV')
    p.add_argument('--unlabeled', nargs='*', default=[], help='Optional unlabeled CSV files')
    p.add_argument('--output', required=True, help='JSON summary path')
    p.add_argument('--model-output', default=None, help='Optional path to save the best trained bundle')
    p.add_argument('--max-unlabeled', type=int, default=0, help='Optional cap on unlabeled rows')
    p.add_argument('--replace-numbers', action='store_true')
    p.add_argument('--include-ssl', action='store_true', help='Include SSL component candidates')
    p.add_argument('--weight-step', type=float, default=0.1)
    p.add_argument('--threshold-step', type=float, default=0.01)
    return p


def _load_unlabeled(paths: list[str], max_rows: int = 0) -> list[str]:
    texts: list[str] = []
    for path in paths:
        for row in read_examples(path):
            texts.append(row.text)
            if max_rows and len(texts) >= max_rows:
                return deduplicate_texts(texts)
    return deduplicate_texts(texts)


def _candidate_grid(replace_numbers: bool):
    vectorizers = [
        VectorizerConfig((1, 2), (3, 5), 5000, 5000, 1, replace_numbers),
        VectorizerConfig((1, 3), (3, 5), 20000, 10000, 1, replace_numbers),
        VectorizerConfig((1, 2), (3, 4), 10000, 5000, 1, replace_numbers),
    ]
    logistics = [
        LogisticConfig(c=0.25, solver='liblinear', penalty='l2'),
        LogisticConfig(c=1.0, solver='liblinear', penalty='l2'),
        LogisticConfig(c=10.0, solver='liblinear', penalty='l2'),
        LogisticConfig(c=100.0, solver='liblinear', penalty='l2'),
        LogisticConfig(c=100.0, solver='saga', penalty='elasticnet', l1_ratio=0.1, max_iter=8000),
    ]
    for vi, vectorizer_cfg in enumerate(vectorizers):
        for li, logistic_cfg in enumerate(logistics):
            yield f'v{vi}_lr{li}', vectorizer_cfg, logistic_cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    labeled = read_labeled_dataset(args.train, max_words=MAX_WORDS, replace_numbers=args.replace_numbers)
    _, texts, labels = zip(*labeled)
    texts = list(texts)
    labels = list(labels)
    unlabeled = _load_unlabeled(args.unlabeled, args.max_unlabeled)
    ssl_cfg = SelfTrainingConfig(enabled=args.include_ssl)

    results = []
    for name, vectorizer_cfg, logistic_cfg in _candidate_grid(args.replace_numbers):
        component_probs = cross_validated_component_probs(
            texts,
            labels,
            unlabeled if unlabeled else None,
            vectorizer_cfg=vectorizer_cfg,
            ssl_cfg=ssl_cfg,
            embedding_cfg=None,
            logistic_cfg=logistic_cfg,
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
        print(name, results[-1]['metrics'], results[-1]['weights'], flush=True)

    results.sort(key=lambda r: (r['metrics']['f1'], r['metrics']['accuracy'], r['metrics']['recall']), reverse=True)
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
            embedding_cfg=None,
            logistic_cfg=LogisticConfig(**best['logistic_cfg']),
            fixed_weights=best['weights'],
            threshold_step=args.threshold_step,
        )
        save_bundle(bundle, args.model_output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
