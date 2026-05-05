from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import MAX_WORDS
from .unlabeled_cache import read_cache
from .text import normalize_for_vectorizer, truncate_words


@dataclass(frozen=True)
class TextExample:
    row_id: str
    text: str
    label: int | None = None


def read_examples(path: str | Path) -> list[TextExample]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2**31 - 1)
        reader = csv.DictReader(f)
        rows: list[TextExample] = []
        for row in reader:
            label = row.get("label")
            rows.append(
                TextExample(
                    row_id=str(row.get("row_id", "")).strip(),
                    text=str(row.get("text", "")),
                    label=None if label in (None, "") else int(label),
                )
            )
    return rows


def split_xy(examples: Iterable[TextExample], max_words: int = MAX_WORDS, replace_numbers: bool = False):
    rows = list(examples)
    texts = [
        truncate_words(normalize_for_vectorizer(row.text, replace_numbers=replace_numbers), max_words=max_words)
        for row in rows
    ]
    labels = [row.label for row in rows]
    row_ids = [row.row_id for row in rows]
    return row_ids, texts, labels


def read_labeled_dataset(path: str | Path, max_words: int = MAX_WORDS, replace_numbers: bool = False):
    row_ids, texts, labels = split_xy(read_examples(path), max_words=max_words, replace_numbers=replace_numbers)
    labeled = [(rid, txt, int(lbl)) for rid, txt, lbl in zip(row_ids, texts, labels) if lbl is not None]
    if not labeled:
        raise ValueError(f"No labels found in {path}")
    return labeled


def read_unlabeled_dataset(path: str | Path, max_words: int = MAX_WORDS, replace_numbers: bool = False):
    row_ids, texts, labels = split_xy(read_examples(path), max_words=max_words, replace_numbers=replace_numbers)
    return [(rid, txt) for rid, txt, lbl in zip(row_ids, texts, labels)]


def read_unlabeled_records(path: str | Path, max_words: int = MAX_WORDS, replace_numbers: bool = False):
    path = Path(path)
    if path.suffix == '.parquet' or path.name.endswith('.parquet.gz'):
        records = read_cache(path)
        return [
            (r.row_id, truncate_words(normalize_for_vectorizer(r.sentence, replace_numbers=replace_numbers), max_words=max_words))
            for r in records
            if r.drop_reason == ''
        ]
    if path.suffix in {'.gz', '.csv'} or path.name.endswith('.csv.gz'):
        row_ids, texts, labels = split_xy(read_examples(path), max_words=max_words, replace_numbers=replace_numbers)
        return [(rid, txt) for rid, txt, lbl in zip(row_ids, texts, labels)]
    return read_unlabeled_dataset(path, max_words=max_words, replace_numbers=replace_numbers)
