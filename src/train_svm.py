"""SVM model training script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_config
from src.contracts import ARTIFACT_KIND_BASELINE
from src.data import get_texts_labels, load_train_data, load_training_manifest
from src.manifest import RunManifest, save_run_manifest
from src.models.svm import SVMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + SVM model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", help="Single training CSV path")
    group.add_argument("--train-manifest", dest="train_manifest", help="Training manifest JSON (multi-shard)")
    parser.add_argument("--val", default=None, help="Optional validation CSV path")
    parser.add_argument("--output", default=None, help="Output model pickle path")
    parser.add_argument("--max_features", type=int, default=None, help="Max TF-IDF features")
    parser.add_argument("--ngram_range", type=int, nargs=2, default=None, help="Example: 1 2")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path for run manifest JSON (default: next to model as <name>_run_manifest.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.train_manifest:
        manifest_path = Path(args.train_manifest)
        if not manifest_path.exists():
            print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)
        train_df = load_training_manifest(manifest_path)
        train_source = str(manifest_path.resolve())
    else:
        train_path = Path(args.train)
        if not train_path.exists():
            print(f"Error: training file not found: {train_path}", file=sys.stderr)
            sys.exit(1)
        train_df = load_train_data(train_path)
        train_source = str(train_path.resolve())

    texts, labels = get_texts_labels(train_df)
    print(f"Loaded {len(texts)} training rows from {train_source}")

    model = SVMModel(
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

    output_path = Path(args.output) if args.output else config.baseline_model_path.parent / "svm_model.pkl"
    model.save(output_path)
    print(f"Saved SVM model to {output_path}")

    manifest_out = Path(args.manifest) if args.manifest else output_path.with_name(output_path.stem + "_run_manifest.json")
    run_manifest = RunManifest(
        backend="svm",
        artifact_kind=ARTIFACT_KIND_BASELINE,
        pretrained_source="tfidf+svm",
        checkpoint_dir=str(output_path.resolve()),
        train_path=train_source,
        max_length=None,
        truncation_policy="ngram_sklearn",
        random_state=config.random_state,
        hyperparams={
            "max_features": args.max_features or config.max_features,
            "ngram_range": list(args.ngram_range) if args.ngram_range else list(config.ngram_range),
            "svm_c": config.logistic_c,
        },
    )
    save_run_manifest(run_manifest, manifest_out)
    print(f"Saved run manifest to {manifest_out}")


if __name__ == "__main__":
    main()
