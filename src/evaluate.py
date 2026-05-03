"""Evaluation script for models on a labeled test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data import get_texts_labels, load_train_data
from src.eval_metrics import binary_classification_metrics
from src.models.baseline import BaselineModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on a labeled test set")
    parser.add_argument("--model", required=True, help="Model path (pickle or directory)")
    parser.add_argument("--input", required=True, help="Labeled test CSV path")
    parser.add_argument(
        "--mode",
        "--backend",
        choices=["baseline", "transformer", "svm"],
        required=True,
        dest="backend",
        help="Model backend (alias: --backend). Public contract: only these values are supported.",
    )
    parser.add_argument("--output", default=None, help="Output JSON path for metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # load_train_data handles labeled CSVs
    df = load_train_data(input_path)
    texts, labels = get_texts_labels(df)

    if args.backend == "baseline":
        model = BaselineModel.load(model_path)
        predictions = model.predict(texts)
    elif args.backend == "svm":
        from src.models.svm import SVMModel
        model = SVMModel.load(model_path)
        predictions = model.predict(texts)
    else:
        from src.models.transformer import TransformerClassifier
        model = TransformerClassifier(model_name=str(model_path))
        predictions = model.predict(texts)

    metrics = binary_classification_metrics(labels, predictions)

    print(f"\n--- Metrics for {args.backend} model on {input_path.name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
