from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .data import read_examples
from .model import load_bundle, predict_component_proba
from .text import has_strong_negation, normalize_for_vectorizer, truncate_words


def build_parser():
    p = argparse.ArgumentParser(description='Predict ICD-codable labels')
    p.add_argument('--model', required=True, help='Path to saved model bundle (.joblib)')
    p.add_argument('--input', required=True, help='Path to input CSV with row_id,text')
    p.add_argument('--output', required=True, help='Path to output CSV')
    p.add_argument('--threshold', type=float, default=None, help='Override model threshold')
    p.add_argument('--component', choices=['ensemble', 'baseline', 'ssl', 'embedding'], default='ensemble', help='Which component to use for prediction')
    p.add_argument('--negation-filter', action='store_true', help='Convert predictions to 0 for sentences with strong negation')
    p.add_argument('--debug-output', default=None, help='Optional CSV with row_id,text,probability,prediction')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    bundle = load_bundle(args.model)
    threshold = bundle.threshold if args.threshold is None else args.threshold
    rows = read_examples(args.input)
    vectorizer_cfg = bundle.metadata.get('vectorizer_cfg', {})
    replace_numbers = bool(vectorizer_cfg.get('replace_numbers', False))
    texts = [truncate_words(normalize_for_vectorizer(r.text, replace_numbers=replace_numbers)) for r in rows]
    row_ids = [r.row_id for r in rows]
    probs = predict_component_proba(bundle, texts, args.component)
    preds = [1 if p >= threshold else 0 for p in probs]
    if args.negation_filter:
        negated = sum(1 for t in texts if has_strong_negation(t))
        preds = [0 if has_strong_negation(t) else p for t, p in zip(texts, preds)]
        print(f'negation filter: {negated}/{len(texts)} predictions flipped')
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['row_id', 'prediction'])
        for row_id, pred in zip(row_ids, preds):
            writer.writerow([row_id, pred])
    if args.debug_output:
        debug_path = Path(args.debug_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['row_id', 'text', 'probability', 'prediction'])
            for row, prob, pred in zip(rows, probs, preds):
                writer.writerow([row.row_id, row.text, float(prob), int(pred)])
        print(f'wrote debug predictions to {debug_path}')
    print(f'wrote {len(preds)} predictions to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
