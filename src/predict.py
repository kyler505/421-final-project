"""Prediction script supporting both baseline and transformer modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_config
from src.data import get_row_ids, get_texts, load_test_data
from src.models.baseline import BaselineModel
from src.utils import save_debug_predictions, save_submission_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument("--model", required=True, help="Model path (pickle or directory)")
    parser.add_argument("--input", required=True, help="Input test CSV path")
    parser.add_argument("--output", default=None, help="Output predictions CSV path")
    parser.add_argument(
        "--mode",
        choices=["baseline", "transformer"],
        required=True,
        help="Model mode",
    )
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length")
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Include positive-class probabilities in a debug CSV",
    )
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Optional debug CSV path with row_id,text,prediction[,probability]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    input_path = Path(args.input)
    model_path = Path(args.model)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    test_df = load_test_data(input_path)
    row_ids = get_row_ids(test_df)
    texts = get_texts(test_df)
    probs = None

    if args.mode == "baseline":
        model = BaselineModel.load(model_path)
        predictions = model.predict(texts)
        if args.probabilities:
            probs = model.predict_proba(texts)[:, 1].tolist()
    else:
        from src.models.transformer import TransformerClassifier

        model = TransformerClassifier(
            model_name=str(model_path),
            max_length=args.max_length or config.max_length,
        )
        predictions = model.predict(texts)
        if args.probabilities:
            probs = model.predict_proba(texts)[:, 1].tolist()

    output_path = Path(args.output) if args.output else config.predictions_path
    save_submission_predictions(row_ids=row_ids, predictions=predictions.tolist(), output_path=output_path)

    if args.debug_output:
        save_debug_predictions(
            row_ids=row_ids,
            texts=texts,
            predictions=predictions.tolist(),
            output_path=args.debug_output,
            probs=probs,
        )

    positives = int(predictions.sum())
    print(f"Saved submission predictions to {output_path}")
    if args.debug_output:
        print(f"Saved debug predictions to {args.debug_output}")
    print(f"Positive predictions: {positives}/{len(predictions)}")


if __name__ == "__main__":
    main()
