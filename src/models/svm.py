"""SVM model with TF-IDF features."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


class SVMModel:
    """SVM text classification baseline."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        c: float = 1.0,
    ) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_df=max_df,
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    SVC(
                        kernel="linear",
                        C=c,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "SVMModel":
        self.pipeline.fit(list(texts), list(labels))
        self._is_fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.pipeline.predict(list(texts))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.pipeline.predict_proba(list(texts))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SVMModel":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
