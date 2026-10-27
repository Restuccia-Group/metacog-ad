"""SuperADD's own scorer behind the :class:`~.base.Scorer` interface, otherwise unchanged.

Both primitives are called from ``industrial.nearest_neighbor``, which is byte-identical to upstream
(``tests/test_core.py`` pins its sha256), so this module adds no distance or subsampling logic.

Two details are easy to misread:

- **Scoring uses k = 1** (``model.py:239``). The paper's "k=100 nearest neighbors" is the bank
  subsampling (``subsampling_distance_based_fast(..., knn_neighbors=100)``), not the scoring.
- **The bank is not a uniform sample.** The subsampling keeps each token with probability inverse
  to its local density, which over-represents sparse regions.

The coreset needs random access to the whole corpus, so this arm keeps every token until
:meth:`finalize`, as upstream does (``model.py:196-202``). For MVTec AD 2 ``walnuts`` that is 59 GB
across the four layers, so :meth:`partial_fit` fails early when the corpus outgrows the RAM budget.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from industrial.nearest_neighbor import nearest_neighbors, subsampling_distance_based_fast

from .base import Scorer

SCORING_K = 1          # model.py:239
SUBSAMPLING_K = 100    # model.py:209

# Refuse a fit whose retained corpus would exceed this fraction of the RAM available at the start.
CORPUS_BUDGET_FRACTION = 0.5


def available_bytes() -> int:
    """``MemAvailable`` from /proc/meminfo."""
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


class KnnScorer(Scorer):
    """A memory bank per layer: an inverse-density coreset of the prototype tokens, scored by 1-NN."""

    kind = "knn"

    def __init__(
        self,
        layers: list[int],
        device: str | torch.device,
        max_database_size: int = 100_000,
        subsampling_iterations: int = 100,
    ) -> None:
        self.layers = list(layers)
        self.device = torch.device(device)
        self.max_database_size = max_database_size
        self.subsampling_iterations = subsampling_iterations
        self._blocks: dict[int, list[np.ndarray]] = {}
        self._bank: dict[int, Tensor] = {}
        self._n_tokens: dict[int, int] = {}
        self._retained_bytes = 0
        self._budget = int(available_bytes() * CORPUS_BUDGET_FRACTION)

    def partial_fit(self, layer: int, block: np.ndarray) -> None:
        block = np.ascontiguousarray(block, dtype=np.float32)
        self._retained_bytes += block.nbytes
        assert self._retained_bytes <= self._budget, (
            f"KNN corpus reached {self._retained_bytes / 1e9:.1f} GB, over the "
            f"{self._budget / 1e9:.1f} GB budget ({CORPUS_BUDGET_FRACTION:.0%} of the RAM available "
            f"at fit start). The coreset needs every token in memory. Run this category alone."
        )
        self._blocks.setdefault(layer, []).append(block)

    def finalize(self) -> None:
        assert self._blocks, "finalize() called before any partial_fit()"
        for layer in sorted(self._blocks):
            # One layer at a time, so peak memory is one corpus plus one bank.
            corpus = np.concatenate(self._blocks.pop(layer), axis=0)
            self._n_tokens[layer] = int(corpus.shape[0])
            bank = subsampling_distance_based_fast(
                corpus,
                self.max_database_size,
                str(self.device),
                iterations=self.subsampling_iterations,
                normalize=False,
                knn_neighbors=SUBSAMPLING_K,
            )
            del corpus
            self._bank[layer] = torch.as_tensor(bank).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self._retained_bytes = 0

    @torch.no_grad()
    def distance(self, layer: int, query: Tensor) -> Tensor:
        bank = self._bank[layer]
        # Upstream's call. With k = 1 the mean is a no-op; it is kept so the line matches model.py:239.
        distances, _ = nearest_neighbors(
            query.to(bank.device), bank, knn_neighbors=SCORING_K, normalize=False
        )
        return distances.mean(dim=-1).to(query.device)

    def model_bytes(self) -> int:
        return sum(bank.element_size() * bank.nelement() for bank in self._bank.values())

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "k": SCORING_K,
            "max_database_size": self.max_database_size,
            "subsampling_k": SUBSAMPLING_K,
            "subsampling_iterations": self.subsampling_iterations,
            "bank_size": {str(k): int(v.shape[0]) for k, v in sorted(self._bank.items())},
            "n_train_tokens": {str(k): v for k, v in sorted(self._n_tokens.items())},
            "model_bytes": self.model_bytes(),
        }
