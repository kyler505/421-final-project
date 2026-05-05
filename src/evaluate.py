from __future__ import annotations

import argparse
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .config import MAX_WORDS
from .data import read_labeled_dataset
from .model import load_bundle, predict_component_proba


def build_parser():
    p = argparse.ArgumentParser(description='Evaluate ICD-codable model')
    p.add_argument('--model', required=True, help='Path to saved model bundle (.joblib)')
    p.add_argument('--data', required=True, help='Path to labeled data CSV')
    p.add_argument('--component', choices=['ensemble', 'baseline', 'ssl', 'embedding'], default='ensemble', help='Which component to score')
    p.add_argument('--all-components', action='store_true', help='Also print component-by-component metrics')
    return p


def _score_component(bundle, texts, labels, component, threshold=None):
    probs = predict_component_proba(bundle, texts, component)
    thresh = bundle.threshold if threshold is None else threshold
    preds = (probs >= thresh).astype(int)
    return {
        'component': component,
        'threshold': float(thresh),
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(labels, preds).tolist(),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    bundle = load_bundle(args.model)
    labeled = read_labeled_dataset(args.data, max_words=MAX_WORDS)
    _, texts, labels = zip(*labeled)
    texts = list(texts)
    labels = list(labels)

    result = _score_component(bundle, texts, labels, args.component)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(classification_report(labels, (predict_component_proba(bundle, texts, args.component) >= result['threshold']).astype(int), digits=4, zero_division=0))

    if args.all_components:
        print('--- all components ---')
        for name in bundle.component_names():
            comp = _score_component(bundle, texts, labels, name)
            print(json.dumps(comp, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
