"""Cross-validation script for Model C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_config
from src.data import get_texts_labels, load_train_data
from src.eval_cv import cv_model_stratified
from src.models.embedding_clf import EmbeddingClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CV for Model C")
    parser.add_argument("--train", required=True, help="Training data CSV path")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Local checkpoint path or model identifier",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--max_length", type=int, default=128, help="Max tokenized length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--clf_type", choices=["logistic", "svm"], default="logistic", help="Classifier type")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--use_tfidf", action="store_true", help="Combine embeddings with TF-IDF features")
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
    
    model_name = args.model_name or config.model_name
    
    print(f"Running {args.n_splits}-fold CV for Model C with {model_name}...")
    
    def model_factory():
        return EmbeddingClassifier(
            model_name=model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            clf_type=args.clf_type,
            use_tfidf=args.use_tfidf,
            c=args.c,
            random_state=config.random_state,
        )
    
    fold_metrics, summary = cv_model_stratified(
        texts=texts,
        labels=labels,
        n_splits=args.n_splits,
        random_state=config.random_state,
        model_factory=model_factory,
    )
    
    print("\nFold Results:")
    for metrics in fold_metrics:
        print(f"Fold {metrics['fold']}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k.capitalize()}: {v:.4f}")


if __name__ == "__main__":
    main()
