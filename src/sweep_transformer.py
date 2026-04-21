"""Small hyperparameter sweep for transformer classification on tiny datasets.

Designed for datasets like the CSCE 421 final project with very few labeled rows.
Uses stratified K-fold CV and a small hand-picked search space.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.data import load_train_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small transformer hyperparameter sweep")
    parser.add_argument("--train", required=True, help="Training CSV path")
    parser.add_argument("--model_name", required=True, help="Local checkpoint path or model id")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument("--workdir", default="sweep_runs", help="Directory for temporary fold checkpoints")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import Trainer, TrainingArguments

    train_df = load_train_data(args.train)
    texts = train_df["text"].tolist()
    labels = train_df["label"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    search_space = [
        {"name": "lr2e-5_ep3_bs4", "learning_rate": 2e-5, "epochs": 3, "batch_size": 4},
        {"name": "lr1e-5_ep3_bs4", "learning_rate": 1e-5, "epochs": 3, "batch_size": 4},
        {"name": "lr3e-5_ep3_bs4", "learning_rate": 3e-5, "epochs": 3, "batch_size": 4},
        {"name": "lr2e-5_ep5_bs4", "learning_rate": 2e-5, "epochs": 5, "batch_size": 4},
        {"name": "lr1e-5_ep5_bs4", "learning_rate": 1e-5, "epochs": 5, "batch_size": 4},
        {"name": "lr2e-5_ep3_bs8", "learning_rate": 2e-5, "epochs": 3, "batch_size": 8},
    ]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    for config in search_space:
        fold_metrics = []
        print(f"=== Running config: {config['name']} ===", flush=True)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
            fold_dir = workdir / config["name"] / f"fold_{fold_idx}"
            if fold_dir.exists():
                shutil.rmtree(fold_dir)
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_batch, batched=True)
            val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize_batch, batched=True)

            model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

            training_args = TrainingArguments(
                output_dir=str(fold_dir),
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["batch_size"],
                num_train_epochs=config["epochs"],
                learning_rate=config["learning_rate"],
                weight_decay=0.01,
                warmup_steps=0,
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="no",
                report_to=[],
                disable_tqdm=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
            )
            trainer.train()

            preds = trainer.predict(val_ds)
            pred_labels = np.argmax(preds.predictions, axis=-1)

            metrics = {
                "fold": fold_idx,
                "accuracy": accuracy_score(val_labels, pred_labels),
                "f1": f1_score(val_labels, pred_labels, zero_division=0),
                "precision": precision_score(val_labels, pred_labels, zero_division=0),
                "recall": recall_score(val_labels, pred_labels, zero_division=0),
            }
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, flush=True)
            fold_metrics.append(metrics)

            shutil.rmtree(fold_dir, ignore_errors=True)

        result = {
            **config,
            "folds": fold_metrics,
            "mean_accuracy": mean(m["accuracy"] for m in fold_metrics),
            "mean_f1": mean(m["f1"] for m in fold_metrics),
            "mean_precision": mean(m["precision"] for m in fold_metrics),
            "mean_recall": mean(m["recall"] for m in fold_metrics),
        }
        all_results.append(result)
        print(
            "SUMMARY",
            result["name"],
            {
                "accuracy": round(result["mean_accuracy"], 4),
                "f1": round(result["mean_f1"], 4),
                "precision": round(result["mean_precision"], 4),
                "recall": round(result["mean_recall"], 4),
            },
            flush=True,
        )

    all_results.sort(key=lambda x: (x["mean_f1"], x["mean_accuracy"]), reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "train_path": str(args.train),
        "model_name": args.model_name,
        "folds": args.folds,
        "max_length": args.max_length,
        "best": all_results[0],
        "results": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved sweep results to {output_path}")
    print("BEST", all_results[0]["name"], {"f1": round(all_results[0]["mean_f1"], 4), "accuracy": round(all_results[0]["mean_accuracy"], 4)}, flush=True)


if __name__ == "__main__":
    main()
