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
from src.data import get_texts_labels, load_train_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer classifier scaffold")
    parser.add_argument("--train", required=True, help="Training data CSV path")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"Error: training file not found: {train_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        print(
            "Error: transformer training requires optional packages. "
            "Install torch/transformers/datasets/accelerate first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    train_df = load_train_data(train_path)
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

    training_args = TrainingArguments(
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print(f"Training on {len(texts)} rows for {epochs} epoch(s)...")
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved transformer scaffold checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
