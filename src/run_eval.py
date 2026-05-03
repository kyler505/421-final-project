"""CLI: stratified CV on gold labels (baseline) — shared metric definitions with sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import get_config
from src.data import get_texts_labels, load_train_data, load_training_manifest
from src.eval_cv import cv_model_stratified
from src.models.baseline import BaselineModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model with stratified K-fold CV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", help="Single labeled training CSV")
    group.add_argument("--train-manifest", dest="train_manifest", help="Training manifest JSON (multi-shard)")
    parser.add_argument(
        "--mode",
        choices=["baseline", "svm"],
        default="baseline",
        help="Model backend to evaluate",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (<= n_samples)")
    parser.add_argument("--random-state", type=int, default=42, dest="random_state")
    parser.add_argument("--output", default=None, help="Optional JSON path for fold metrics + means")
    parser.add_argument("--max-features", type=int, default=None, dest="max_features")
    parser.add_argument("--ngram-range", type=int, nargs=2, default=None, dest="ngram_range")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.train_manifest:
        train_df = load_training_manifest(Path(args.train_manifest))
        source = Path(args.train_manifest)
    else:
        train_path = Path(args.train)
        if not train_path.exists():
            print(f"Error: training file not found: {train_path}", file=sys.stderr)
            sys.exit(1)
        train_df = load_train_data(train_path)
        source = train_path

    texts, labels = get_texts_labels(train_df)
    n_samples = len(texts)
    if args.folds > n_samples:
        print(f"Error: folds ({args.folds}) > n_samples ({n_samples})", file=sys.stderr)
        sys.exit(1)

    ngram = tuple(args.ngram_range) if args.ngram_range else config.ngram_range
    max_feat = args.max_features or config.max_features

    if args.mode == "baseline":
        def factory() -> BaselineModel:
            return BaselineModel(
                max_features=max_feat,
                ngram_range=ngram,
                min_df=config.min_df,
                max_df=config.max_df,
                c=config.logistic_c,
            )
    elif args.mode == "svm":
        from src.models.svm import SVMModel
        def factory() -> SVMModel:
            return SVMModel(
                max_features=max_feat,
                ngram_range=ngram,
                min_df=config.min_df,
                max_df=config.max_df,
                c=config.logistic_c,
            )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    fold_rows, means = cv_model_stratified(
        texts,
        labels,
        n_splits=args.folds,
        random_state=args.random_state,
        model_factory=factory,
    )

    print(f"CV source: {source} (mode: {args.mode})")
    for row in fold_rows:
        fold = int(row["fold"])
        parts = {k: row[k] for k in ("accuracy", "f1", "precision", "recall")}
        print(f"fold {fold}: " + ", ".join(f"{k}={parts[k]:.4f}" for k in parts))
    print("MEAN:", ", ".join(f"{k}={v:.4f}" for k, v in means.items()))

    if args.output:
        out = {
            "train_source": str(source),
            "folds": args.folds,
            "per_fold": fold_rows,
            "mean": means,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
