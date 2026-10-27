"""The interface between SuperADD's pipeline and the model of what a normal patch looks like.

Upstream hardcodes a memory bank: ``SuperADD.train`` collects the tokens, coresets them, and
``predict`` scores by 1-NN (``model.py:196-211``, ``:238-241``). Everything around those two blocks
is independent of the scorer, and is what the comparison holds fixed.

A scorer owns the **accumulation**, not only the fit. The kNN arm must keep every token until it
can coreset them, while the Mahalanobis arm folds each block into a ``[d, d]`` Gram matrix and
drops it. A ``fit(all_tokens)`` interface would force both arms to hold the whole corpus, which is
59 GB for MVTec AD 2 ``walnuts``.

``partial_fit`` takes the ``[T, C]`` float32 numpy array upstream already builds at
``model.py:202``. ``distance`` returns ``[T]`` on the device of its input, before the ``/ c`` and
the reshape, which is what ``nearest_neighbors(...).mean(dim=-1)`` returned at ``model.py:240``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor


class Scorer(ABC):
    """A per-layer model of normal tokens. One instance holds every layer."""

    #: The registry key, e.g. ``"knn"`` or ``"mahalanobis"``.
    kind: str = ""

    @abstractmethod
    def partial_fit(self, layer: int, block: np.ndarray) -> None:
        """Absorb one image's ``[T, C]`` float32 tokens for one layer."""

    @abstractmethod
    def finalize(self) -> None:
        """Close the fit, once, after the last :meth:`partial_fit`. The expensive work goes here."""

    @abstractmethod
    def distance(self, layer: int, query: Tensor) -> Tensor:
        """``[T, C]`` -> ``[T]``, larger is more anomalous, on the device of ``query``."""

    @abstractmethod
    def model_bytes(self) -> int:
        """Size of the fitted model: what has to be kept to run inference."""

    @abstractmethod
    def summary(self) -> dict:
        """Diagnostics for the results JSON. Never read by the scoring."""
