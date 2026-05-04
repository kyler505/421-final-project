"""Transformer-based feature extraction for clinical notes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch


class TransformerEmbedder:
    """Extracts fixed-length embeddings from a transformer model (frozen)."""

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
        device: str | None = None,
        pooling: str = "mean",  # "mean", "cls", or "max"
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or self._resolve_device()
        self.pooling = pooling
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
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Transformer embedding requires 'torch' and 'transformers'. "
                "Install optional dependencies before using this path."
            ) from exc

        load_from = str(Path(self.model_name)) if Path(self.model_name).exists() else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        self.model = AutoModel.from_pretrained(load_from)
        self.model.to(self.device)
        self.model.eval()

    def _iter_batches(self, texts: Sequence[str], batch_size: int = 16):
        text_list = list(texts)
        for start in range(0, len(text_list), batch_size):
            yield text_list[start : start + batch_size]

    def embed(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        self._lazy_load()
        import torch

        embeddings = []
        with torch.no_grad():
            for batch in self._iter_batches(texts, batch_size=batch_size):
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                
                # Use the last hidden state
                last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
                
                if self.pooling == "cls":
                    # Take the [CLS] token (index 0)
                    batch_embeddings = last_hidden_state[:, 0, :]
                elif self.pooling == "mean":
                    # Mean pooling over non-padding tokens
                    attention_mask = encoded["attention_mask"]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                elif self.pooling == "max":
                    # Max pooling over non-padding tokens
                    attention_mask = encoded["attention_mask"]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding to large negative value
                    batch_embeddings = torch.max(last_hidden_state, 1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")
                
                embeddings.append(batch_embeddings.cpu().numpy())
                
        return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 768), dtype=float)
