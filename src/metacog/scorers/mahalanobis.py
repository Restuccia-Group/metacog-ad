"""One Gaussian per layer, fitted on every prototype token, scored by whitened distance.

This arm changes one thing in SuperADD: the definition of "far from normal". The downscale, tiling,
brightness augmentation, prototype split, ``/ c``, layer mean, threshold and morphology are all
upstream's. The fit is global: one mean and one covariance per layer, pooled over every position of
every prototype image.

**The score is** ``sqrt(d^2 / c)`` **rather than** ``d^2``. SuperADD's threshold is a
multiplicative gain, ``1.421 x p95`` (``model.py:218``), tuned on L2 nearest-neighbour distances.
On a squared distance the same gain is only ``sqrt(1.421) = 1.19x`` in distance terms, a much
tighter cut. The square root makes the map an L2 distance in whitened space, so the gain keeps its
meaning.

The ``/ c`` goes inside the root because ``d^2`` under the fitted Gaussian is roughly chi-square
with ``c`` degrees of freedom. ``sqrt(d^2 / c)`` then has expectation near 1 at every layer
whatever its width, so the four layers enter the mean at ``model.py:248`` on equal terms, which is
the job upstream's ``/ c`` does for kNN. Both transforms are monotone, so they cannot change a
per-layer ranking, only the cross-layer mean and the thresholds.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .base import Scorer
from .covariance import COV_DTYPE, StreamingCovariance, precision_from_covariance

# Query rows scored at once, to bound the float64 [rows, C] intermediates (84 MB at C = 1280).
SCORE_CHUNK_ROWS = 8192


class MahalanobisScorer(Scorer):
    """A Gaussian per layer. The fit is one streaming pass; the model is a mean and a precision."""

    kind = "mahalanobis"

    def __init__(self, layers: list[int], device: str | torch.device) -> None:
        self.layers = list(layers)
        self.device = torch.device(device)
        self._accumulators: dict[int, StreamingCovariance] = {}
        self._mean: dict[int, Tensor] = {}
        self._precision: dict[int, Tensor] = {}
        self._n_tokens: dict[int, int] = {}
        self._trace: dict[int, float] = {}
        self._top_eigenvalues: dict[int, list[float]] = {}
        self._bottom_eigenvalues: dict[int, list[float]] = {}

    def partial_fit(self, layer: int, block: np.ndarray) -> None:
        tokens = torch.as_tensor(block)
        if layer not in self._accumulators:
            self._accumulators[layer] = StreamingCovariance(int(tokens.shape[1]), self.device)
        self._accumulators[layer].update(tokens)

    def finalize(self) -> None:
        assert self._accumulators, "finalize() called before any partial_fit()"
        for layer, accumulator in self._accumulators.items():
            mean, cov = accumulator.finalize()
            self._mean[layer] = mean
            self._precision[layer] = precision_from_covariance(cov)
            self._n_tokens[layer] = accumulator.n
            # Diagnostics for the results JSON. The smallest eigenvalues get the largest weights
            # in the precision, so they show whether the inverse can be trusted.
            eigenvalues = torch.linalg.eigvalsh(cov).flip(0)
            self._trace[layer] = float(torch.diagonal(cov).sum())
            self._top_eigenvalues[layer] = [float(v) for v in eigenvalues[:5]]
            self._bottom_eigenvalues[layer] = [float(v) for v in eigenvalues[-5:].flip(0)]
        self._accumulators.clear()

    @torch.no_grad()
    def distance(self, layer: int, query: Tensor) -> Tensor:
        assert self._mean, "distance() called before finalize()"
        mean = self._mean[layer].to(query.device)
        precision = self._precision[layer].to(query.device)
        c = int(query.shape[1])

        out = torch.empty(query.shape[0], dtype=torch.float32, device=query.device)
        for start in range(0, query.shape[0], SCORE_CHUNK_ROWS):
            chunk = query[start : start + SCORE_CHUNK_ROWS].to(COV_DTYPE) - mean
            squared = torch.einsum("nd,de,ne->n", chunk, precision, chunk)
            out[start : start + SCORE_CHUNK_ROWS] = torch.clamp(squared, min=0.0).div_(c).sqrt_().float()
        return out

    def model_bytes(self) -> int:
        tensors = [*self._mean.values(), *self._precision.values()]
        return sum(t.element_size() * t.nelement() for t in tensors)

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "fit": "global",
            "map": "sqrt(mahalanobis^2 / c)",
            "n_train_tokens": {str(k): v for k, v in sorted(self._n_tokens.items())},
            "model_bytes": self.model_bytes(),
            "trace_per_layer": {str(k): v for k, v in sorted(self._trace.items())},
            "top_eigenvalues": {str(k): v for k, v in sorted(self._top_eigenvalues.items())},
            # Ascending: the five directions the precision weights most.
            "bottom_eigenvalues": {str(k): v for k, v in sorted(self._bottom_eigenvalues.items())},
            "top_eigenvalue_share": {
                str(k): sum(v) / self._trace[k]
                for k, v in sorted(self._top_eigenvalues.items()) if self._trace.get(k)
            },
        }
