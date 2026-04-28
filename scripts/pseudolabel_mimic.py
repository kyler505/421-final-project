#!/usr/bin/env python3
"""Pseudolabel MIMIC-III NOTEEVENTS using a trained baseline model.

Workflow:
  1. Load MIMIC NOTEEVENTS (gzip or plain CSV). Filter by CATEGORY if provided.
  2. Chunk each note into 128-word fragments, sentence-aware.
  3. Run baseline model.predict_proba() on all chunks in batches.
  4. Keep only high-confidence predictions (--confidence, default 0.95).
  5. Write silver CSV: row_id, text, label.
  6. Write manifest combining gold CSV + silver CSV.

Dependencies: pandas, scikit-learn (already in requirements.txt).
Baseline model must already exist at --baseline-path.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import get_config
from src.models.baseline import BaselineModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudolabel MIMIC-III using baseline model")
    parser.add_argument(
        "--mimic-csv",
        required=True,
        help="Path to NOTEEVENTS.csv or NOTEEVENTS.csv.gz",
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
        help="Path to baseline_model.pkl (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write pseudolabels CSV + manifest (default: data/processed)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence threshold for keeping pseudolabels (default 0.95)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Max words per chunk (default from config: max_words_course=128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size",
    )
    parser.add_argument(
        "--categories",
        default="Discharge summary",
        help="Comma-separated list of NOTEEVENTS CATEGORY values to include (default: 'Discharge summary')",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process first N notes (for quick prototyping)",
    )
    parser.add_argument(
        "--gold-csv",
        default=None,
        help="Path to gold training CSV for manifest (default: data/raw/train_data-text_and_labels.csv)",
    )
    return parser.parse_args()


def sentence_split(text: str) -> list[str]:
    """Very light sentence splitter without NLTK dependency."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_note(text: str, max_words: int) -> list[str]:
    """Chunk text into fragments of up to max_words each, preferring sentence boundaries."""
    sentences = sentence_split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        # Flush current chunk if adding this sentence would exceed the limit
        if current and (current_words + sent_words > max_words):
            chunks.append(" ".join(current))
            current = [sent]
            current_words = sent_words
        else:
            current.append(sent)
            current_words += sent_words

        # If a single sentence exceeds max_words, force-split it into word windows
        if sent_words > max_words:
            words = sent.split()
            sub_chunks = [
                " ".join(words[i : i + max_words])
                for i in range(0, len(words), max_words)
            ]
            # Replace the long sentence with sub-chunks (merged forward when possible)
            if current and current[-1] == sent:
                current.pop()
                current_words -= sent_words
                for sc in sub_chunks:
                    sc_words = len(sc.split())
                    if current and (current_words + sc_words <= max_words):
                        current.append(sc)
                        current_words += sc_words
                    else:
                        if current:
                            chunks.append(" ".join(current))
                        current = [sc]
                        current_words = sc_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def load_mimic_notes(
    mimic_csv: str | Path,
    categories: set[str],
    sample_n: int | None = None,
) -> pd.DataFrame:
    """Load NOTEEVENTS, return DataFrame with ROW_ID, CATEGORY, TEXT."""
    path = Path(mimic_csv)
    if path.suffix == ".gz":
        df = pd.read_csv(path, compression="gzip", low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)

    required = {"ROW_ID", "CATEGORY", "TEXT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NOTEEVENTS missing required columns: {missing}")

    # Filter categories
    df = df[df["CATEGORY"].isin(categories)].copy()
    if sample_n is not None:
        df = df.head(sample_n).copy()
    return df[["ROW_ID", "CATEGORY", "TEXT"]].reset_index(drop=True)


def run_inference(
    model: BaselineModel,
    chunks: list[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, probabilities) for all chunks."""
    preds_list: list[int] = []
    probs_list: list[float] = []
    total = len(chunks)
    for i in tqdm(range(0, total, batch_size), desc="Inferring", unit="batch"):
        batch = chunks[i : i + batch_size]
        batch_preds = model.predict(batch)
        batch_probs = model.predict_proba(batch)[:, 1]  # P(class=1)
        preds_list.extend(batch_preds.tolist())
        probs_list.extend(batch_probs.tolist())
    return np.array(preds_list), np.array(probs_list)


def write_silver_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    """Write pseudolabel CSV with row_id, text, label."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} silver rows to {output_path}")


def make_relative_path(path: Path, base: Path) -> str:
    """Convert absolute path to relative-to-base string for manifest portability."""
    try:
        rel = path.resolve().relative_to(base.resolve())
        return str(rel)
    except ValueError:
        # Not relative — fall back to absolute
        return str(path.resolve())


def write_manifest(
    manifest_path: Path,
    gold_path: Path | None,
    silver_paths: list[Path],
    baseline_path: Path,
    confidence: float,
    total_rows: int,
) -> None:
    """Create training manifest for gold + silver shards with relative paths."""
    base_dir = manifest_path.parent
    entries: list[dict] = []

    if gold_path is not None:
        entries.append({
            "path": make_relative_path(gold_path, base_dir),
            "label_source": "gold_human",
            "split": "gold_train",
        })
    for sp in silver_paths:
        entries.append({
            "path": make_relative_path(sp, base_dir),
            "label_source": f"silver_pseudolabel_conf_{confidence:.2f}",
            "split": "silver_train",
        })

    payload = {
        "version": 1,
        "entries": entries,
        "baseline_model": make_relative_path(baseline_path, base_dir),
        "total_rows": total_rows,
        "notes": f"Generated by pseudolabel_mimic.py (conf={confidence})",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote manifest with {len(entries)} entries to {manifest_path}")


def main() -> None:
    args = parse_args()
    config = get_config()

    # Paths
    baseline_path = Path(args.baseline_path) if args.baseline_path else config.baseline_model_path
    if not baseline_path.exists():
        print(f"Error: baseline model not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("data/processed")
    max_words = args.max_words or config.max_words_course

    # Parse categories
    categories = {c.strip() for c in args.categories.split(",")}
    print(f"Categories filter: {categories}")

    # 1) Load MIMIC notes
    print(f"Loading MIMIC notes from {args.mimic_csv} …")
    mimic_df = load_mimic_notes(args.mimic_csv, categories, sample_n=args.sample)
    print(f"Loaded {len(mimic_df)} notes after category filter")

    # 2) Chunk
    all_chunks: list[str] = []
    chunk_meta: list[dict] = []   # stores (original_row_id, chunk_index) per chunk
    for _, row in tqdm(mimic_df.iterrows(), total=len(mimic_df), desc="Chunking"):
        note_id = int(row["ROW_ID"])
        text = str(row["TEXT"])
        chunks = chunk_note(text, max_words)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_meta.append({"note_id": note_id, "chunk_idx": idx})
    print(f"Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks produced — exiting.", file=sys.stderr)
        sys.exit(1)

    # 3) Load baseline model
    print(f"Loading baseline model from {baseline_path} …")
    model = BaselineModel.load(baseline_path)

    # 4) Inference
    preds, probs = run_inference(model, all_chunks, args.batch_size)

    # 5) Confidence filter
    pos_mask = (probs >= args.confidence) & (preds == 1)
    neg_mask = ((1 - probs) >= args.confidence) & (preds == 0)
    keep_mask = pos_mask | neg_mask
    kept_indices = keep_mask.nonzero()[0]
    print(f"Kept {len(kept_indices)}/{len(all_chunks)} high-confidence chunks "
          f"({len(kept_indices)/len(all_chunks)*100:.1f}%)")

    # 6) Write silver CSV (single file) with integer row_ids
    silver_rows = []
    row_id_counter = 1  # global sequential counter; avoids string IDs
    for idx in kept_indices:
        meta = chunk_meta[idx]
        silver_rows.append({
            "row_id": row_id_counter,
            "text": all_chunks[idx],
            "label": int(preds[idx]),
        })
        row_id_counter += 1

    silver_path = output_dir / "pseudolabels.csv"
    write_silver_csv(silver_path, silver_rows)

    # Label distribution sanity check
    labels = [r["label"] for r in silver_rows]
    from collections import Counter
    dist = Counter(labels)
    print(f"Silver label distribution: {dict(dist)}")

    # 7) Write manifest
    gold_path = Path(args.gold_csv) if args.gold_csv else Path("data/raw/train_data-text_and_labels.csv")
    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path=manifest_path,
        gold_path=gold_path if gold_path.exists() else None,
        silver_paths=[silver_path],
        baseline_path=baseline_path,
        confidence=args.confidence,
        total_rows=len(silver_rows),
    )

    print("Pseudolabeling complete.")
    print(f"Next: train combined baseline with:")
    print(f"  python -m src.train_baseline --train-manifest {manifest_path} --output models/baseline_model_combined.pkl")


if __name__ == "__main__":
    main()
