"""Dataset loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import Config, get_config


def infer_column(df: pd.DataFrame, preferred_names: tuple[str, ...] | list[str]) -> Optional[str]:
    """Infer a column name from common variations."""
    for name in preferred_names:
        if name in df.columns:
            return name
    lowered = {column.lower().strip(): column for column in df.columns}
    for name in preferred_names:
        match = lowered.get(name.lower().strip())
        if match is not None:
            return match
    return None


def load_csv(
    path: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    row_id_column: str | None = None,
    has_labels: bool = True,
    config: Config | None = None,
) -> pd.DataFrame:
    """Load a project CSV and normalize column names."""
    config = config or get_config()

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    text_column = text_column or infer_column(df, config.text_inference_columns)
    row_id_column = row_id_column or infer_column(df, config.row_id_inference_columns)
    if has_labels:
        label_column = label_column or infer_column(df, config.label_inference_columns)

    if text_column is None:
        raise ValueError(f"Could not infer text column from {df.columns.tolist()}")

    result = pd.DataFrame()
    if row_id_column is not None:
        result[config.row_id_column] = df[row_id_column].values
    else:
        result[config.row_id_column] = range(len(df))
    result[config.text_column] = df[text_column].fillna("").astype(str).values

    if has_labels:
        if label_column is None:
            raise ValueError(f"Could not infer label column from {df.columns.tolist()}")
        result[config.label_column] = df[label_column].astype(int).values

    return result


def load_train_data(
    path: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    row_id_column: str | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    return load_csv(
        path=path,
        text_column=text_column,
        label_column=label_column,
        row_id_column=row_id_column,
        has_labels=True,
        config=config,
    )


def load_test_data(
    path: str | Path,
    text_column: str | None = None,
    row_id_column: str | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    return load_csv(
        path=path,
        text_column=text_column,
        row_id_column=row_id_column,
        has_labels=False,
        config=config,
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    return train_test_split(df, test_size=test_size, random_state=random_state)


def get_texts_labels(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    return df["text"].tolist(), df["label"].tolist()


def get_texts(df: pd.DataFrame) -> list[str]:
    return df["text"].tolist()


def get_row_ids(df: pd.DataFrame) -> list[int]:
    return df["row_id"].tolist()


def load_training_manifest(path: str | Path, config: Config | None = None) -> pd.DataFrame:
    """Load and concatenate labeled shards described by a JSON manifest.

    Manifest schema: see docs/data-manifest-schema.md. Each entry must point to a
    CSV loadable by load_train_data. Adds optional columns ``label_source`` and
    ``split`` for provenance (gold vs silver, etc.).
    """
    config = config or get_config()
    path = Path(path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported training manifest version: {version}")

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a non-empty 'entries' list")

    base = path.parent
    frames: list[pd.DataFrame] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"Manifest entries[{idx}] must be an object with 'path'")
        rel = Path(str(entry["path"]))
        csv_path = rel if rel.is_absolute() else (base / rel).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Manifest shard not found: {csv_path}")

        df = load_train_data(csv_path, config=config)
        df = df.copy()
        df["label_source"] = str(entry.get("label_source", "unknown"))
        df["split"] = str(entry.get("split", "none"))
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
