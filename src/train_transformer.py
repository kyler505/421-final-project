"""Transformer training entrypoint for the project scaffold.

This is intentionally conservative: it wires together tokenization, dataset
construction, and a Hugging Face Trainer setup, but leaves room for later
cross-validation, custom metrics, and stronger experiment tracking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_config
from src.contracts import ARTIFACT_KIND_TRANSFORMER
from src.data import get_texts_labels, load_train_data, load_training_manifest
from src.eval_metrics import binary_classification_metrics
from src.manifest import RunManifest, save_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer classifier scaffold")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", help="Single training data CSV path")
    group.add_argument("--train-manifest", dest="train_manifest", help="Training manifest JSON (multi-shard)")
    parser.add_argument("--output", default=None, help="Output model directory")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Local checkpoint path or model identifier (prefer local path for offline use)",
    )
    parser.add_argument("--max_length", type=int, default=None, help="Max tokenized length")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--val", default=None, help="Optional validation CSV path for best-checkpoint selection")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience when validation is enabled")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional path for run manifest JSON (default: <output_dir>/run_manifest.json)",
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

    try:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import Trainer, TrainingArguments
    except ImportError:
        print(
            "Error: transformer training requires optional packages. "
            "Install torch/transformers/datasets/accelerate first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    texts, labels = get_texts_labels(train_df)

    model_name = args.model_name or config.model_name
    max_length = args.max_length or config.max_length
    batch_size = args.batch_size or config.batch_size
    epochs = args.epochs or config.epochs
    learning_rate = args.learning_rate or config.learning_rate
    output_dir = Path(args.output) if args.output else config.transformer_model_path

    print(f"Loading tokenizer/model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize_batch, batched=True)

    eval_df = None
    eval_source = None
    if args.val:
        val_path = Path(args.val)
        if not val_path.exists():
            print(f"Error: validation file not found: {val_path}", file=sys.stderr)
            sys.exit(1)
        eval_df = load_train_data(val_path)
        eval_source = str(val_path.resolve())

    eval_dataset = None
    if eval_df is not None:
        eval_texts, eval_labels = get_texts_labels(eval_df)
        eval_dataset = Dataset.from_dict({"text": eval_texts, "label": eval_labels}).map(tokenize_batch, batched=True)

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, metric_labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return binary_classification_metrics(metric_labels, predictions)

    training_args_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    if eval_dataset is not None:
        training_args_kwargs.update(
            dict(
                eval_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                save_total_limit=1,
            )
        )
    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    if eval_dataset is not None:
        trainer_kwargs.update(
            dict(
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
            )
        )
        try:
            from transformers import EarlyStoppingCallback

            trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
        except Exception:
            pass
    trainer = Trainer(**trainer_kwargs)

    if eval_source:
        print(
            f"Training on {len(texts)} rows from {train_source} for {epochs} epoch(s) with validation on {eval_source}..."
        )
    else:
        print(f"Training on {len(texts)} rows from {train_source} for {epochs} epoch(s)...")
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved transformer scaffold checkpoint to {output_dir}")

    manifest_out = Path(args.manifest) if args.manifest else (output_dir / "run_manifest.json")
    run_manifest = RunManifest(
        backend="transformer",
        artifact_kind=ARTIFACT_KIND_TRANSFORMER,
        pretrained_source=str(model_name),
        checkpoint_dir=str(output_dir.resolve()),
        train_path=train_source,
        max_length=max_length,
        truncation_policy="hf_max_length_tokens",
        random_state=config.random_state,
        hyperparams={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": config.weight_decay,
            "max_words_course": config.max_words_course,
            "validation_path": eval_source,
            "early_stopping_patience": args.patience,
        },
    )
    save_run_manifest(run_manifest, manifest_out)
    print(f"Saved run manifest to {manifest_out}")


if __name__ == "__main__":
    main()
