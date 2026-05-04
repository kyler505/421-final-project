"""Small hyperparameter sweep for TF-IDF linear text classifiers.

This is intentionally tiny and transparent: it runs stratified CV over a handful
of hand-picked configurations for either the logistic-regression baseline or the
LinearSVC baseline.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from src.data import get_texts_labels, load_train_data, load_training_manifest
from src.eval_cv import cv_model_stratified
from src.models.baseline import BaselineModel
from src.models.svm import SVMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small sweep for linear text models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", help="Single training CSV path")
    group.add_argument("--train-manifest", dest="train_manifest", help="Training manifest JSON (multi-shard)")
    parser.add_argument("--backend", choices=["baseline", "svm"], default="baseline")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified folds")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument("--random_state", type=int, default=42, help="CV random seed")
    return parser.parse_args()


def make_search_space(backend: str) -> list[dict[str, object]]:
    if backend == "baseline":
        max_features = [5000, 10000, 20000]
        ngram_ranges = [(1, 1), (1, 2), (1, 3)]
        c_values = [0.25, 0.5, 1.0, 2.0]
    else:
        max_features = [5000, 10000, 20000]
        ngram_ranges = [(1, 1), (1, 2), (1, 3)]
        c_values = [0.25, 0.5, 1.0, 2.0]

    # Keep the sweep intentionally small and sensible.
    configs = []
    for max_feat, ngram_range, c in product(max_features, ngram_ranges, c_values):
        if backend == "baseline":
            configs.append(
                {
                    "name": f"mf{max_feat}_ng{ngram_range[0]}{ngram_range[1]}_c{c}",
                    "max_features": max_feat,
                    "ngram_range": ngram_range,
                    "c": c,
                    "sublinear_tf": True,
                }
            )
        else:
            configs.append(
                {
                    "name": f"mf{max_feat}_ng{ngram_range[0]}{ngram_range[1]}_c{c}",
                    "max_features": max_feat,
                    "ngram_range": ngram_range,
                    "c": c,
                    "sublinear_tf": True,
                }
            )
    # Prefer the most standard settings early in the list.
    configs.sort(key=lambda x: (x["max_features"], x["ngram_range"], x["c"]))
    return configs[:8]


def build_factory(backend: str, config: dict[str, object]):
    if backend == "baseline":
        return lambda: BaselineModel(
            max_features=int(config["max_features"]),
            ngram_range=tuple(config["ngram_range"]),
            min_df=1,
            max_df=1.0,
            sublinear_tf=bool(config["sublinear_tf"]),
            c=float(config["c"]),
        )
    return lambda: SVMModel(
        max_features=int(config["max_features"]),
        ngram_range=tuple(config["ngram_range"]),
        min_df=1,
        max_df=1.0,
        sublinear_tf=bool(config["sublinear_tf"]),
        c=float(config["c"]),
    )


def main() -> None:
    args = parse_args()

    if args.train_manifest:
        manifest_path = Path(args.train_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        train_df = load_training_manifest(manifest_path)
        train_source = str(manifest_path.resolve())
    else:
        train_path = Path(args.train)
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        train_df = load_train_data(train_path)
        train_source = str(train_path.resolve())

    texts, labels = get_texts_labels(train_df)
    search_space = make_search_space(args.backend)
    all_results: list[dict[str, object]] = []

    for config in search_space:
        folds, means = cv_model_stratified(
            texts,
            labels,
            n_splits=args.folds,
            random_state=args.random_state,
            model_factory=build_factory(args.backend, config),
        )
        result = {
            **config,
            "folds": folds,
            "mean_accuracy": means["accuracy"],
            "mean_f1": means["f1"],
            "mean_precision": means["precision"],
            "mean_recall": means["recall"],
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
    best = all_results[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "backend": args.backend,
        "train_path": train_source,
        "folds": args.folds,
        "best": best,
        "results": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"Saved sweep results to {output_path}",
        {"best": best["name"], "f1": round(best["mean_f1"], 4), "accuracy": round(best["mean_accuracy"], 4)},
        flush=True,
    )


if __name__ == "__main__":
    main()
