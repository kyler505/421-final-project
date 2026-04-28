"""Shared binary classification metrics for baselines, transformers, and sweeps."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Return a flat dict suitable for JSON logs and sweeps."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
    }
