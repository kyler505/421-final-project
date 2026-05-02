"""
Changing these constants or validation rules is a breaking change for teammates
and any automation; extend via new optional fields rather than renaming.
"""

from __future__ import annotations

from typing import Sequence

# Course / Gradescope submission CSV
SUBMISSION_HEADER: tuple[str, str] = ("row_id", "prediction")
PREDICT_BACKENDS: tuple[str, str, str] = ("baseline", "transformer", "svm")
ARTIFACT_KIND_BASELINE: str = "sklearn_pickle"
ARTIFACT_KIND_TRANSFORMER: str = "huggingface_directory"


def validate_submission_rows(row_ids: Sequence[int], predictions: Sequence[int]) -> None:
    """Ensure submission arrays are parallel and predictions are binary {0, 1}."""
    row_list = list(row_ids)
    pred_list = list(predictions)
    if len(row_list) != len(pred_list):
        raise ValueError(f"row_ids length {len(row_list)} != predictions length {len(pred_list)}")
    for i, p in enumerate(pred_list):
        if int(p) not in (0, 1):
            raise ValueError(f"prediction at index {i} must be 0 or 1, got {p!r}")


def validate_submission_csv_header(header_row: list[str]) -> None:
    expected = list(SUBMISSION_HEADER)
    if [c.strip() for c in header_row] != expected:
        raise ValueError(f"submission header must be {expected}, got {header_row!r}")
