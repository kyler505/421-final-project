from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .text import normalize_for_vectorizer, normalize_text, truncate_words

_SHORT_REASONS = {'too_short', 'too_long', 'low_alpha_ratio', 'boilerplate', 'duplicate'}
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r'^\s*electronically signed by\b',
        r'^\s*discharge instructions\b',
        r'^\s*please call your doctor\b',
        r'^\s*follow up as needed\b',
        r'^\s*sign(?:ed)? by\b',
        r'^\s*phone number\b',
        r'^\s*patient instructions\b',
    ]
]


@dataclass(frozen=True)
class UnlabeledCandidate:
    row_id: str
    note_id: str
    sentence: str
    sentence_hash: str
    word_count: int
    section_header: str = ''
    drop_reason: str = ''


def sentence_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode('utf-8')).hexdigest()


def _looks_boilerplate(text: str) -> bool:
    return any(p.search(text) for p in _BOILERPLATE_PATTERNS)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text))


def _filter_reason(sentence: str, word_count: int, seen: set[str]) -> str:
    norm = normalize_text(sentence)
    if not sentence or word_count < 4:
        return 'too_short'
    if word_count > 128:
        return 'too_long'
    if _alpha_ratio(sentence) < 0.35:
        return 'low_alpha_ratio'
    if _looks_boilerplate(sentence):
        return 'boilerplate'
    if norm in seen:
        return 'duplicate'
    return ''


def filter_candidates(
    rows: Iterable[UnlabeledCandidate],
    per_note_cap: int = 25,
    candidate_cap: int = 0,
) -> tuple[list[UnlabeledCandidate], list[UnlabeledCandidate]]:
    kept: list[UnlabeledCandidate] = []
    dropped: list[UnlabeledCandidate] = []
    seen: set[str] = set()
    per_note_counts: dict[str, int] = {}
    for row in rows:
        if per_note_cap and per_note_counts.get(row.note_id, 0) >= per_note_cap:
            dropped.append(UnlabeledCandidate(**{**asdict(row), 'drop_reason': 'per_note_cap'}))
            continue
        reason = _filter_reason(row.sentence, row.word_count, seen)
        if reason:
            dropped.append(UnlabeledCandidate(**{**asdict(row), 'drop_reason': reason}))
            continue
        seen.add(normalize_text(row.sentence))
        per_note_counts[row.note_id] = per_note_counts.get(row.note_id, 0) + 1
        kept.append(row)
        if candidate_cap and len(kept) >= candidate_cap:
            break
    return kept, dropped


def write_cache(rows: list[UnlabeledCandidate], output: str | Path) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    if out.suffix == '.parquet':
        try:
            df.to_parquet(out, index=False)
            return
        except Exception:
            pass
        out = out.with_suffix('.csv.gz')
    if out.suffix.endswith('.gz'):
        df.to_csv(out, index=False, compression='gzip')
    else:
        df.to_csv(out, index=False)


def read_cache(path: str | Path) -> list[UnlabeledCandidate]:
    p = Path(path)
    if p.suffix == '.parquet':
        try:
            df = pd.read_parquet(p)
        except Exception:
            alt = p.with_suffix('.csv.gz')
            df = pd.read_csv(alt, compression='gzip')
    else:
        df = pd.read_csv(p)
    rows: list[UnlabeledCandidate] = []
    for rec in df.to_dict(orient='records'):
        rows.append(UnlabeledCandidate(
            row_id=str(rec.get('row_id', '')),
            note_id=str(rec.get('note_id', '')),
            sentence=str(rec.get('sentence', rec.get('text', ''))),
            sentence_hash=str(rec.get('sentence_hash', sentence_hash(str(rec.get('sentence', rec.get('text', '')))))),
            word_count=int(rec.get('word_count', len(str(rec.get('sentence', rec.get('text', ''))).split()))),
            section_header=str(rec.get('section_header', '')),
            drop_reason=str(rec.get('drop_reason', '')),
        ))
    return rows
