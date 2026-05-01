#!/usr/bin/env python3
"""Pseudolabel MIMIC-III NOTEEVENTS using a teacher model.

Workflow:
  1. Load MIMIC NOTEEVENTS (gzip or plain CSV). Filter by CATEGORY if provided.
  2. Chunk each note into 128-word fragments, sentence-aware.
  3. Run a teacher model on all chunks in batches.
     - baseline: TF-IDF + Logistic Regression
     - transformer: fine-tuned sequence classifier checkpoint
  4. Keep high-confidence predictions, but fall back to class-balanced top-k
     selection if the confidence threshold yields too few silver rows.
  5. Write silver CSV: row_id, text, label (+ provenance columns).
  6. Write manifest combining gold CSV + silver CSV.

Dependencies: pandas, scikit-learn, and optionally torch/transformers when using
``--teacher-mode transformer``.
The baseline teacher model must already exist at --baseline-path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import get_config
from src.data import load_train_data
from src.models.baseline import BaselineModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudolabel MIMIC-III using teacher model")
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
        "--teacher-mode",
        choices=("baseline", "transformer"),
        default="baseline",
        help="Teacher backend to use for pseudolabeling",
    )
    parser.add_argument(
        "--teacher-path",
        default=None,
        help="Path to a fine-tuned transformer checkpoint (required for teacher-mode=transformer)",
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=None,
        help="Optional batch size override for transformer teacher inference",
    )
    parser.add_argument(
        "--teacher-max-length",
        type=int,
        default=None,
        help="Optional token length override for transformer teacher inference",
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
        default=0.90,
        help="High-confidence threshold for keeping pseudolabels (default 0.90)",
    )
    parser.add_argument(
        "--min-silver-rows",
        type=int,
        default=1000,
        help="Minimum total silver rows to keep after fallback selection",
    )
    parser.add_argument(
        "--min-silver-fraction",
        type=float,
        default=0.01,
        help="Minimum silver fraction of total chunks to keep after fallback selection",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=250,
        help="Minimum rows to retain per class when falling back to top-k selection",
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
        help="Comma-separated list of NOTEEVENTS CATEGORY values to include (empty string disables filtering)",
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
        # Flush current chunk if adding this sentence would exceed the limit.
        if current and (current_words + sent_words > max_words):
            chunks.append(" ".join(current))
            current = [sent]
            current_words = sent_words
        else:
            current.append(sent)
            current_words += sent_words

        # If a single sentence exceeds max_words, force-split it into word windows.
        if sent_words > max_words:
            words = sent.split()
            sub_chunks = [
                " ".join(words[i : i + max_words])
                for i in range(0, len(words), max_words)
            ]
            # Replace the long sentence with sub-chunks (merged forward when possible).
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


def infer_column_name(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name, case-insensitively."""
    for name in candidates:
        if name in df.columns:
            return name
    lowered = {column.lower().strip(): column for column in df.columns}
    for name in candidates:
        match = lowered.get(name.lower().strip())
        if match is not None:
            return match
    return None


def normalize_categories(raw: str) -> set[str]:
    if not raw or not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


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

    df.columns = df.columns.str.strip()
    row_id_col = infer_column_name(df, ("ROW_ID", "row_id"))
    category_col = infer_column_name(df, ("CATEGORY", "category"))
    text_col = infer_column_name(df, ("TEXT", "text"))

    missing = [name for name, col in (("ROW_ID", row_id_col), ("CATEGORY", category_col), ("TEXT", text_col)) if col is None]
    if missing:
        raise ValueError(f"NOTEEVENTS missing required columns: {missing}; found {df.columns.tolist()}")

    df = df[[row_id_col, category_col, text_col]].rename(
        columns={row_id_col: "ROW_ID", category_col: "CATEGORY", text_col: "TEXT"}
    )
    df["CATEGORY"] = df["CATEGORY"].fillna("").astype(str).str.strip()
    df["TEXT"] = df["TEXT"].fillna("").astype(str)

    if categories:
        df = df[df["CATEGORY"].isin(categories)].copy()
    if sample_n is not None:
        df = df.head(sample_n).copy()
    return df.reset_index(drop=True)


def run_baseline_inference(
    model: BaselineModel,
    chunks: list[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, probabilities) for a baseline teacher."""
    preds_list: list[int] = []
    probs_list: list[float] = []
    total = len(chunks)
    for i in tqdm(range(0, total, batch_size), desc="Inferring (baseline)", unit="batch"):
        batch = chunks[i : i + batch_size]
        batch_preds = model.predict(batch)
        batch_probs = model.predict_proba(batch)[:, 1]  # P(class=1)
        preds_list.extend(batch_preds.tolist())
        probs_list.extend(batch_probs.tolist())
    return np.array(preds_list), np.array(probs_list)


def load_transformer_teacher(teacher_path: str | Path):
    """Load a sequence-classification checkpoint for teacher inference."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformer teacher mode requires torch and transformers to be installed"
        ) from exc

    teacher_path = Path(teacher_path)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Transformer teacher checkpoint not found: {teacher_path}")

    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    model = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def run_transformer_inference(
    teacher_path: str | Path,
    chunks: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, probabilities) for a transformer teacher."""
    tokenizer, model, device = load_transformer_teacher(teacher_path)

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Transformer teacher mode requires torch to be installed") from exc

    preds_list: list[int] = []
    probs_list: list[float] = []
    total = len(chunks)
    for i in tqdm(range(0, total, batch_size), desc="Inferring (transformer)", unit="batch"):
        batch = chunks[i : i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
                batch_probs = probs
                batch_preds = (probs >= 0.5).long()
            else:
                probs = torch.softmax(logits, dim=-1)
                batch_probs = probs[:, 1]
                batch_preds = torch.argmax(probs, dim=-1)
        preds_list.extend(batch_preds.detach().cpu().tolist())
        probs_list.extend(batch_probs.detach().cpu().tolist())
    return np.array(preds_list), np.array(probs_list)


def select_silver_examples(
    probs: np.ndarray,
    confidence: float,
    min_silver_rows: int,
    min_silver_fraction: float,
    min_per_class: int,
) -> tuple[list[dict], dict[str, int]]:
    """Select pseudo-labels using thresholding with class-balanced fallback.

    Returns a list of selection records with:
      - idx: original chunk index
      - label: pseudo-label (0/1)
      - confidence: selection confidence for the chosen label
      - reason: threshold / class_fallback / fill
    """
    total_chunks = len(probs)
    if total_chunks == 0:
        return [], {"target_rows": 0, "target_per_class": 0}

    target_rows = min(
        total_chunks,
        max(min_silver_rows, int(round(total_chunks * min_silver_fraction))),
    )
    target_per_class = max(min_per_class, math.ceil(target_rows / 2))

    pos_order = np.argsort(-probs)
    neg_order = np.argsort(probs)

    selected: dict[int, dict[str, float | int | str]] = {}
    class_counts = {0: 0, 1: 0}

    def add(idx: int, label: int, confidence_score: float, reason: str) -> bool:
        if idx in selected:
            return False
        selected[idx] = {
            "idx": idx,
            "label": label,
            "confidence": confidence_score,
            "reason": reason,
        }
        class_counts[label] += 1
        return True

    # Primary rule: keep only truly confident examples.
    for idx in np.where(probs >= confidence)[0]:
        add(int(idx), 1, float(probs[idx]), "threshold")
    for idx in np.where(probs <= 1.0 - confidence)[0]:
        add(int(idx), 0, float(1.0 - probs[idx]), "threshold")

    # Fallback 1: ensure each class has at least some silver rows.
    for label, order in ((1, pos_order), (0, neg_order)):
        for idx in order:
            if class_counts[label] >= target_per_class:
                break
            idx = int(idx)
            if idx in selected:
                continue
            if label == 1:
                add(idx, 1, float(probs[idx]), "class_fallback")
            else:
                add(idx, 0, float(1.0 - probs[idx]), "class_fallback")

    # Fallback 2: if we still don't have enough rows, fill by highest certainty
    # while keeping the label balance roughly even.
    if len(selected) < target_rows:
        remaining_pos = [int(idx) for idx in pos_order if int(idx) not in selected]
        remaining_neg = [int(idx) for idx in neg_order if int(idx) not in selected]
        pos_i = 0
        neg_i = 0

        while len(selected) < target_rows and (pos_i < len(remaining_pos) or neg_i < len(remaining_neg)):
            choose_label = 1 if class_counts[1] <= class_counts[0] else 0
            if choose_label == 1 and pos_i < len(remaining_pos):
                idx = remaining_pos[pos_i]
                pos_i += 1
                add(idx, 1, float(probs[idx]), "fill")
            elif choose_label == 0 and neg_i < len(remaining_neg):
                idx = remaining_neg[neg_i]
                neg_i += 1
                add(idx, 0, float(1.0 - probs[idx]), "fill")
            elif pos_i < len(remaining_pos):
                idx = remaining_pos[pos_i]
                pos_i += 1
                add(idx, 1, float(probs[idx]), "fill")
            elif neg_i < len(remaining_neg):
                idx = remaining_neg[neg_i]
                neg_i += 1
                add(idx, 0, float(1.0 - probs[idx]), "fill")
            else:
                break

    selected_records = list(selected.values())
    selected_records.sort(key=lambda r: (-float(r["confidence"]), int(r["label"]), int(r["idx"])))
    selection_stats = {
        "target_rows": target_rows,
        "target_per_class": target_per_class,
        "selected_rows": len(selected_records),
        "selected_pos": class_counts[1],
        "selected_neg": class_counts[0],
    }
    return selected_records, selection_stats


def write_silver_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    """Write pseudolabel CSV with row_id, text, label (+ provenance)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "text",
                "label",
                "confidence",
                "source_row_id",
                "chunk_idx",
                "selection_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} silver rows to {output_path}")


def make_relative_path(path: Path, base: Path) -> str:
    """Convert absolute path to relative-to-base string for manifest portability."""
    try:
        rel = path.resolve().relative_to(base.resolve())
        return str(rel)
    except ValueError:
        # Not relative — fall back to absolute.
        return str(path.resolve())


def write_manifest(
    manifest_path: Path,
    gold_path: Path | None,
    silver_paths: list[Path],
    teacher_path: Path,
    teacher_mode: str,
    confidence: float,
    total_rows: int,
    notes: str,
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
        "baseline_model": make_relative_path(teacher_path, base_dir),
        "teacher_mode": teacher_mode,
        "teacher_model": make_relative_path(teacher_path, base_dir),
        "total_rows": total_rows,
        "notes": notes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote manifest with {len(entries)} entries to {manifest_path}")


def get_max_row_id(gold_path: Path) -> int:
    """Return the maximum row_id in the gold CSV, or 0 if unavailable."""
    if not gold_path.exists():
        return 0
    gold_df = load_train_data(gold_path)
    row_ids = pd.to_numeric(gold_df["row_id"], errors="coerce").dropna()
    if row_ids.empty:
        return int(len(gold_df))
    return int(row_ids.max())


def resolve_gold_csv(explicit_path: str | None, config_train_csv: Path) -> Path | None:
    """Choose the gold CSV used in the combined manifest.

    Preference order:
      1. explicit --gold-csv if provided
      2. config.train_csv (project default)
      3. legacy train_data-text_and_labels.csv path
    """
    if explicit_path:
        candidate = Path(explicit_path)
        return candidate if candidate.exists() else None

    candidates = [
        config_train_csv,
        Path("data/raw/train_data-text_and_labels.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    config = get_config()

    # Paths.
    baseline_path = Path(args.baseline_path) if args.baseline_path else config.baseline_model_path
    teacher_mode = args.teacher_mode
    if teacher_mode == "baseline" and not baseline_path.exists():
        print(f"Error: baseline model not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path("data/processed")
    max_words = args.max_words or config.max_words_course
    categories = normalize_categories(args.categories)
    print(f"Categories filter: {categories if categories else '<disabled>'}")

    if teacher_mode == "baseline":
        teacher_path = baseline_path
        teacher_batch_size = args.teacher_batch_size or args.batch_size
        teacher_max_length = args.teacher_max_length or config.max_length
    else:
        teacher_path = Path(args.teacher_path) if args.teacher_path else config.transformer_model_path
        teacher_batch_size = args.teacher_batch_size or args.batch_size
        teacher_max_length = args.teacher_max_length or config.max_length
        if not teacher_path.exists():
            print(
                f"Error: transformer teacher checkpoint not found: {teacher_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Teacher mode: {teacher_mode} ({teacher_path})")

    # 1) Load MIMIC notes.
    print(f"Loading MIMIC notes from {args.mimic_csv} …")
    mimic_df = load_mimic_notes(args.mimic_csv, categories, sample_n=args.sample)
    print(f"Loaded {len(mimic_df)} notes after category filter")

    # 2) Chunk.
    all_chunks: list[str] = []
    chunk_meta: list[dict] = []
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

    # 3) Load teacher model and infer.
    if teacher_mode == "baseline":
        print(f"Loading baseline teacher from {teacher_path} …")
        model = BaselineModel.load(teacher_path)
        preds, probs = run_baseline_inference(model, all_chunks, teacher_batch_size)
    else:
        print(f"Loading transformer teacher from {teacher_path} …")
        preds, probs = run_transformer_inference(
            teacher_path=teacher_path,
            chunks=all_chunks,
            batch_size=teacher_batch_size,
            max_length=teacher_max_length,
        )

    # 4) High-confidence selection with class-balanced fallback.
    selected_records, selection_stats = select_silver_examples(
        probs=probs,
        confidence=args.confidence,
        min_silver_rows=args.min_silver_rows,
        min_silver_fraction=args.min_silver_fraction,
        min_per_class=args.min_per_class,
    )
    print(
        "Selection stats: "
        f"target_rows={selection_stats['target_rows']}, "
        f"target_per_class={selection_stats['target_per_class']}, "
        f"selected={selection_stats['selected_rows']} "
        f"(pos={selection_stats['selected_pos']}, neg={selection_stats['selected_neg']})"
    )

    # 6) Write silver CSV with sequential integer row_ids.
    gold_path = resolve_gold_csv(args.gold_csv, config.train_csv)
    row_id_offset = get_max_row_id(gold_path) if gold_path is not None else 0
    silver_rows: list[dict] = []
    for seq, record in enumerate(selected_records, start=1):
        idx = int(record["idx"])
        label = int(record["label"])
        meta = chunk_meta[idx]
        silver_rows.append(
            {
                "row_id": row_id_offset + seq,
                "text": all_chunks[idx],
                "label": label,
                "confidence": round(float(record["confidence"]), 6),
                "source_row_id": meta["note_id"],
                "chunk_idx": meta["chunk_idx"],
                "selection_reason": record["reason"],
            }
        )

    silver_path = output_dir / "pseudolabels.csv"
    write_silver_csv(silver_path, silver_rows)

    # Label distribution sanity check.
    labels = [r["label"] for r in silver_rows]
    dist = Counter(labels)
    print(f"Silver label distribution: {dict(dist)}")

    # 7) Write manifest.
    manifest_path = output_dir / "manifest.json"
    notes = (
        f"Generated by pseudolabel_mimic.py (teacher_mode={teacher_mode}, teacher={teacher_path}, "
        f"conf={args.confidence}, min_rows={args.min_silver_rows}, "
        f"min_fraction={args.min_silver_fraction}, min_per_class={args.min_per_class})"
    )
    write_manifest(
        manifest_path=manifest_path,
        gold_path=gold_path if gold_path.exists() else None,
        silver_paths=[silver_path],
        teacher_path=teacher_path,
        teacher_mode=teacher_mode,
        confidence=args.confidence,
        total_rows=len(silver_rows),
        notes=notes,
    )

    print("Pseudolabeling complete.")
    print("Next: train combined baseline with:")
    print(f"  python -m src.train_baseline --train-manifest {manifest_path} --output models/baseline_model_combined.pkl")


if __name__ == "__main__":
    main()
