"""Training entrypoint for Model C: Frozen Transformer Embeddings + Classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_config
from src.contracts import ARTIFACT_KIND_BASELINE  # Using pickle for the classifier
from src.data import get_texts_labels, load_train_data, load_training_manifest
from src.manifest import RunManifest, save_run_manifest
from src.models.embedding_clf import EmbeddingClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen transformer embedding classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", help="Single training data CSV path")
    group.add_argument("--train-manifest", dest="train_manifest", help="Training manifest JSON")
    parser.add_argument("--output", required=True, help="Output model pickle path")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Local checkpoint path or model identifier",
    )
    parser.add_argument("--max_length", type=int, default=128, help="Max tokenized length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding extraction")
    parser.add_argument("--clf_type", choices=["logistic", "svm"], default="logistic", help="Classifier type")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--use_tfidf", action="store_true", help="Combine embeddings with TF-IDF features")
    parser.add_argument("--tfidf_max_features", type=int, default=5000, help="Max TF-IDF features")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path for run manifest JSON",
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
    
    model_name = args.model_name or config.model_name
    output_path = Path(args.output)
    
    print(f"Training Model C with {model_name} as feature extractor...")
    print(f"Classifier type: {args.clf_type}, C: {args.c}, TF-IDF: {args.use_tfidf}")
    
    model = EmbeddingClassifier(
        model_name=model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        clf_type=args.clf_type,
        use_tfidf=args.use_tfidf,
        tfidf_max_features=args.tfidf_max_features,
        c=args.c,
        random_state=config.random_state,
    )
    
    model.fit(texts, labels)
    model.save(output_path)
    print(f"Saved Model C to {output_path}")
    
    manifest_out = Path(args.manifest) if args.manifest else output_path.with_suffix(".json")
    if manifest_out == output_path:
        manifest_out = output_path.parent / (output_path.name + "_manifest.json")
        
    run_manifest = RunManifest(
        backend="embedding_clf",
        artifact_kind=ARTIFACT_KIND_BASELINE,
        pretrained_source=str(model_name),
        checkpoint_dir=str(output_path.resolve()),
        train_path=train_source,
        max_length=args.max_length,
        truncation_policy="hf_max_length_tokens",
        random_state=config.random_state,
        hyperparams={
            "clf_type": args.clf_type,
            "c": args.c,
            "use_tfidf": args.use_tfidf,
            "tfidf_max_features": args.tfidf_max_features,
            "batch_size": args.batch_size,
        },
    )
    save_run_manifest(run_manifest, manifest_out)
    print(f"Saved run manifest to {manifest_out}")


if __name__ == "__main__":
    main()
