"""Baseline model training script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_config
from src.data import get_texts_labels, load_train_data
from src.models.baseline import BaselineModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline")
    parser.add_argument("--train", required=True, help="Training data CSV path")
    parser.add_argument("--val", default=None, help="Optional validation CSV path")
    parser.add_argument("--output", default=None, help="Output model pickle path")
    parser.add_argument("--max_features", type=int, default=None, help="Max TF-IDF features")
    parser.add_argument("--ngram_range", type=int, nargs=2, default=None, help="Example: 1 2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"Error: training file not found: {train_path}", file=sys.stderr)
        sys.exit(1)

    train_df = load_train_data(train_path)
    texts, labels = get_texts_labels(train_df)
    print(f"Loaded {len(texts)} training rows from {train_path}")

    model = BaselineModel(
        max_features=args.max_features or config.max_features,
        ngram_range=tuple(args.ngram_range) if args.ngram_range else config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        c=config.logistic_c,
    )
    model.fit(texts, labels)

    if args.val:
        val_df = load_train_data(args.val)
        val_texts, val_labels = get_texts_labels(val_df)
        preds = model.predict(val_texts)
        accuracy = float((preds == val_labels).mean())
        print(f"Validation accuracy: {accuracy:.4f}")

    output_path = Path(args.output) if args.output else config.baseline_model_path
    model.save(output_path)
    print(f"Saved baseline model to {output_path}")


if __name__ == "__main__":
    main()
