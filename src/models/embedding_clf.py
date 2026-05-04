"""Classifier using frozen transformer embeddings + optionally TF-IDF."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from src.models.embedding import TransformerEmbedder


class TransformerEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for extracting transformer embeddings."""

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
        batch_size: int = 16,
        pooling: str = "mean",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.device = device
        self.embedder = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.embedder is None:
            self.embedder = TransformerEmbedder(
                model_name=self.model_name,
                max_length=self.max_length,
                device=self.device,
                pooling=self.pooling,
            )
        return self.embedder.embed(X, batch_size=self.batch_size)


class EmbeddingClassifier:
    """Classifier that uses frozen transformer embeddings and optionally TF-IDF."""

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
        batch_size: int = 16,
        clf_type: str = "logistic",  # "logistic" or "svm"
        use_tfidf: bool = False,
        tfidf_max_features: int = 5000,
        c: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_tfidf = use_tfidf
        
        features = [
            (
                "transformer",
                TransformerEmbeddingTransformer(
                    model_name=model_name,
                    max_length=max_length,
                    batch_size=batch_size,
                ),
            )
        ]
        
        if use_tfidf:
            features.append(
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=tfidf_max_features,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                )
            )
            
        feature_union = FeatureUnion(features)
        
        if clf_type == "logistic":
            clf = LogisticRegression(
                max_iter=1000,
                C=c,
                class_weight="balanced",
                random_state=random_state,
            )
        elif clf_type == "svm":
            clf = LinearSVC(
                C=c,
                class_weight="balanced",
                random_state=random_state,
                dual=False,
            )
        else:
            raise ValueError(f"Unknown clf_type: {clf_type}")

        self.pipeline = Pipeline(
            [
                ("features", feature_union),
                ("scaler", StandardScaler(with_mean=False)),  # Safe for both dense and sparse
                ("clf", clf),
            ]
        )
        self._is_fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "EmbeddingClassifier":
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
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            return self.pipeline.predict_proba(list(texts))
        else:
            # For LinearSVC, we can use decision_function and sigmoid
            scores = self.pipeline.decision_function(list(texts))
            scores = np.asarray(scores, dtype=float)
            if scores.ndim == 1:
                probs_pos = 1.0 / (1.0 + np.exp(-scores))
                probs_neg = 1.0 - probs_pos
                return np.column_stack([probs_neg, probs_pos])
            return scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Note: We don't want to pickle the whole transformer model if possible,
        # but for simplicity we'll pickle the pipeline. 
        # TransformerEmbedder's lazy loading helps avoid pickling the torch model
        # if it hasn't been loaded yet, or if we clear it.
        if "embedder" in self.pipeline.named_steps:
            self.pipeline.named_steps["embedder"].embedder = None
            
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingClassifier":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
