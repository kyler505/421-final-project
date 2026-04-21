"""Configuration defaults for the project scaffold."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"


@dataclass
class Config:
    train_csv: Path = DATA_DIR / "raw" / "train.csv"
    val_csv: Path = DATA_DIR / "raw" / "val.csv"
    test_csv: Path = DATA_DIR / "raw" / "test.csv"

    baseline_model_path: Path = MODELS_DIR / "baseline_model.pkl"
    transformer_model_path: Path = MODELS_DIR / "transformer_model"
    predictions_path: Path = OUTPUTS_DIR / "predictions.csv"

    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    logistic_c: float = 1.0

    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 128
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 0
    weight_decay: float = 0.01

    row_id_column: str = "row_id"
    text_column: str = "text"
    label_column: str = "label"
    row_id_inference_columns: tuple[str, ...] = ("row_id", "id", "rowid")
    text_inference_columns: tuple[str, ...] = (
        "text",
        "sentence",
        "note",
        "clinical_note",
        "utterance",
    )
    label_inference_columns: tuple[str, ...] = (
        "label",
        "target",
        "y",
        "class",
        "icd_codable",
    )

    test_size: float = 0.2
    random_state: int = 42


def get_config() -> Config:
    config = Config()
    if os.getenv("BASELINE_MODEL_PATH"):
        config.baseline_model_path = Path(os.environ["BASELINE_MODEL_PATH"])
    if os.getenv("TRANSFORMER_MODEL_PATH"):
        config.transformer_model_path = Path(os.environ["TRANSFORMER_MODEL_PATH"])
    return config
