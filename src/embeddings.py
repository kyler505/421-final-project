from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EmbeddingEncoderConfig:
    model_name: str
    max_length: int = 256
    batch_size: int = 16
    local_files_only: bool = True


class EmbeddingEncoder:
    def __init__(self, config: EmbeddingEncoderConfig):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'Embedding support requires torch and transformers. '
                'Install requirements.txt or train with --no-embeddings.'
            ) from exc
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
        ).to(self.device)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, 'hidden_size', 0) or 0)
        if self.hidden_size <= 0:
            raise RuntimeError(f'Could not determine hidden size for {config.model_name}')

    def encode(self, texts: Iterable[str], batch_size: int | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        batch_size = batch_size or self.config.batch_size
        chunks: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt',
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                last_hidden = outputs.last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1).to(last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(chunks)


@lru_cache(maxsize=4)
def get_embedding_encoder(model_name: str, max_length: int = 256, batch_size: int = 16, local_files_only: bool = True) -> EmbeddingEncoder:
    return EmbeddingEncoder(
        EmbeddingEncoderConfig(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            local_files_only=local_files_only,
        )
    )
