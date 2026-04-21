"""Transformer model wrapper for the ClinicalBERT scaffold.

This file is intentionally import-safe when torch/transformers are absent.
Heavy dependencies are only imported when transformer functionality is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch


class TransformerClassifier:
    """Thin wrapper around a Hugging Face sequence classifier."""

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or self._resolve_device()
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _lazy_load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Transformer mode requires 'torch' and 'transformers'. "
                "Install optional dependencies before using this path."
            ) from exc

        load_from = str(Path(self.model_name)) if Path(self.model_name).exists() else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_from, num_labels=2)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        self._lazy_load()
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            return torch.argmax(logits, dim=-1).cpu().numpy()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        self._lazy_load()
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def save(self, path: str | Path) -> None:
        self._lazy_load()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerClassifier":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        instance = cls(model_name=str(path))
        instance._lazy_load()
        return instance

    def get_model(self):
        self._lazy_load()
        return self.model

    def get_tokenizer(self):
        self._lazy_load()
        return self.tokenizer
