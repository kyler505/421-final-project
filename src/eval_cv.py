"""Stratified cross-validation helpers for small gold sets."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.eval_metrics import binary_classification_metrics
from src.models.baseline import BaselineModel


def cv_baseline_stratified(
    texts: Sequence[str],
    labels: Sequence[int],
    n_splits: int,
    random_state: int = 42,
    baseline_factory: Callable[[], BaselineModel] | None = None,
) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    """Fit a fresh baseline per fold; return per-fold metrics and means."""
    text_list = list(texts)
    label_list = [int(x) for x in labels]
    factory = baseline_factory or BaselineModel

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows: list[dict[str, float | int]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(text_list, label_list), start=1):
        train_texts = [text_list[i] for i in train_idx]
        train_labels = [label_list[i] for i in train_idx]
        val_texts = [text_list[i] for i in val_idx]
        val_labels = [label_list[i] for i in val_idx]

        model = factory()
        model.fit(train_texts, train_labels)
        preds = model.predict(val_texts)
        metrics = binary_classification_metrics(val_labels, preds)
        metrics["fold"] = fold_idx
        fold_rows.append(metrics)

    keys = ("accuracy", "f1", "precision", "recall")
    means = {k: float(np.mean([row[k] for row in fold_rows])) for k in keys}
    return fold_rows, means
