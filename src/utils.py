"""General utilities for the project scaffold."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from src.contracts import SUBMISSION_HEADER, validate_submission_rows


def save_submission_predictions(
    row_ids: Sequence[int],
    predictions: Sequence[int],
    output_path: str | Path,
) -> None:
    """Save predictions in the course submission format: row_id,prediction."""
    validate_submission_rows(row_ids, predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(SUBMISSION_HEADER))
        for row_id, pred in zip(row_ids, predictions):
            writer.writerow([int(row_id), int(pred)])


def save_debug_predictions(
    row_ids: Sequence[int],
    texts: Sequence[str],
    predictions: Sequence[int],
    output_path: str | Path,
    probs: Sequence[float] | None = None,
) -> None:
    """Save a richer local-debug CSV with ids, text, predictions, and optional probabilities."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["row_id", "text", "prediction"]
    if probs is not None:
        header.append("probability")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for idx, (row_id, text, pred) in enumerate(zip(row_ids, texts, predictions)):
            row: list[Any] = [int(row_id), text, int(pred)]
            if probs is not None:
                row.append(float(probs[idx]))
            writer.writerow(row)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
