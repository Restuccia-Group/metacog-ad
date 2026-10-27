"""The scorer registry. ``model.py`` uses only :func:`build_scorer`."""
from __future__ import annotations

import torch

from .base import Scorer
from .knn import KnnScorer
from .mahalanobis import MahalanobisScorer

SCORERS: dict[str, type[Scorer]] = {cls.kind: cls for cls in (KnnScorer, MahalanobisScorer)}


def build_scorer(kind: str, layers: list[int], device: str | torch.device, **config) -> Scorer:
    assert kind in SCORERS, f"unknown scorer {kind!r}; available: {sorted(SCORERS)}"
    return SCORERS[kind](layers, device, **config)


__all__ = ["Scorer", "KnnScorer", "MahalanobisScorer", "SCORERS", "build_scorer"]
