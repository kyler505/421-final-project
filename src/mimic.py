from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path
from typing import Iterator

from .text import normalize_for_vectorizer, truncate_words

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def iter_gz_csv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    path = Path(path)
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def iter_discharge_summaries(notes_gz_path: str | Path) -> Iterator[dict[str, str]]:
    for row in iter_gz_csv_rows(notes_gz_path):
        category = (row.get('CATEGORY') or '').strip().lower()
        if category == 'discharge summary':
            yield row


def split_candidate_sentences(text: str, max_words: int = 128, min_words: int = 3) -> list[str]:
    text = normalize_for_vectorizer(text)
    pieces = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    out: list[str] = []
    for piece in pieces:
        piece = truncate_words(piece, max_words=max_words)
        if len(piece.split()) >= min_words:
            out.append(piece)
    return out


def iter_sentence_candidates(notes_gz_path: str | Path, max_words: int = 128) -> Iterator[tuple[str, str]]:
    for row in iter_discharge_summaries(notes_gz_path):
        text = row.get('TEXT') or ''
        note_id = row.get('ROW_ID') or ''
        for sent in split_candidate_sentences(text, max_words=max_words):
            yield note_id, sent
