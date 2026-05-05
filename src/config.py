from __future__ import annotations

from dataclasses import dataclass

MAX_WORDS = 128
RANDOM_STATE = 42
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
DEFAULT_THRESHOLD = 0.5
DEFAULT_EMBEDDING_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'


@dataclass(frozen=True)
class VectorizerConfig:
    word_ngram_range: tuple[int, int] = (1, 3)
    char_ngram_range: tuple[int, int] = (3, 5)
    max_features_word: int = 40000
    max_features_char: int = 30000
    min_df: int = 1


@dataclass(frozen=True)
class SelfTrainingConfig:
    enabled: bool = True
    rounds: int = 2
    positive_confidence: float = 0.90
    negative_confidence: float = 0.10
    pseudo_weight: float = 0.35
    # Ranking mode: use top-K instead of absolute thresholds (for tiny datasets)
    rank_mode: bool = True
    rank_top_k: int = 20
    max_pool: int = 500


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    max_length: int = 256
    batch_size: int = 16
    local_files_only: bool = True
