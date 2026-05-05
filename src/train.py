from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from .config import EmbeddingConfig, MAX_WORDS, SelfTrainingConfig, VectorizerConfig
from .data import read_examples, read_labeled_dataset
from .model import fit_full_pipeline, predict_component_proba, save_bundle
from .text import deduplicate_texts


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
    return p


def _load_unlabeled(paths: list[str], max_rows: int = 0) -> list[str]:
    texts: list[str] = []
    for path in paths:
        for row in read_examples(path):
            texts.append(row.text)
            if max_rows and len(texts) >= max_rows:
                return deduplicate_texts(texts)
    return deduplicate_texts(texts)


def _print_component_metrics(bundle, texts, labels):
    for component in bundle.component_names():
        probs = predict_component_proba(bundle, texts, component)
        preds = (probs >= bundle.threshold).astype(int)
        print(f'[{component}] threshold={bundle.threshold:.3f} f1={f1_score(labels, preds, zero_division=0):.4f} acc={accuracy_score(labels, preds):.4f} precision={precision_score(labels, preds, zero_division=0):.4f} recall={recall_score(labels, preds, zero_division=0):.4f}')


def main(argv=None):
    args = build_parser().parse_args(argv)
    labeled = read_labeled_dataset(args.train, max_words=MAX_WORDS)
    row_ids, texts, labels = zip(*labeled)
    unlabeled_texts = _load_unlabeled(args.unlabeled, max_rows=args.max_unlabeled)

    ssl_cfg = SelfTrainingConfig(enabled=not args.no_self_training)
    embedding_cfg = None if args.no_embeddings else EmbeddingConfig(model_name=args.embedding_model)
    bundle, report = fit_full_pipeline(
        list(texts),
        list(labels),
        unlabeled_texts=unlabeled_texts if unlabeled_texts else None,
        vectorizer_cfg=VectorizerConfig(),
        ssl_cfg=ssl_cfg,
        embedding_cfg=embedding_cfg,
        weight_step=args.weight_step,
        threshold_step=args.threshold_step,
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
