"""Model wrappers.

Avoid importing optional heavy transformer dependencies at package import time.
"""

from src.models.baseline import BaselineModel

__all__ = ["BaselineModel"]
