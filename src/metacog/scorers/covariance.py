"""Mean and covariance of the training tokens, in one streaming pass.

The Mahalanobis arm fits one Gaussian to every prototype token of a category, which for MVTec AD 2
``walnuts`` is 378 images x 7,600 tokens x 1,280 dims: 14.7 GB per layer in float32. A two-pass
fit (the mean first, then the centered moments) cannot be used here even at that cost, because
``PreProcessing`` (``model.py:35``) draws a random brightness factor for every training image, so
a second pass through the encoder would see different pixels from the first.

The fit therefore accumulates, for ``y = x - shift``,

    n,  S1 = sum y,  G = sum y y^T

and returns ``mean = shift + S1 / n`` and ``cov = (G - n m m^T) / n`` with ``m = S1 / n``. That is
13 MB per layer at d = 1280. ``shift`` is the first block's mean: it keeps ``y`` near the origin,
so ``G - n m m^T`` does not cancel two large numbers and lose precision.

The covariance is the plain sample covariance divided by ``n``, the same as sklearn's
``empirical_covariance``, which ``tests/test_core.py`` checks it against.
"""
from __future__ import annotations

import torch
from torch import Tensor

# The centered second moment is a difference of large sums over millions of tokens, and float32
# accumulation drifts visibly. The [d, d] update in float64 is still far cheaper than the encoder.
COV_DTYPE = torch.float64


class StreamingCovariance:
    """Mean and covariance from uncentered ``[m, d]`` blocks, accumulated on ``device``."""

    def __init__(self, d: int, device: torch.device) -> None:
        assert d >= 1, f"dimension must be positive, got {d}"
        self.d = d
        self.device = device
        self.n = 0
        self.shift: Tensor | None = None
        self.sum = torch.zeros(d, dtype=COV_DTYPE, device=device)
        self.gram = torch.zeros((d, d), dtype=COV_DTYPE, device=device)

    def update(self, block: Tensor) -> None:
        """Fold in one ``[m, d]`` block of raw samples."""
        assert block.ndim == 2 and block.shape[1] == self.d, (
            f"expected [m, {self.d}], got {tuple(block.shape)}"
        )
        x = block.to(dtype=COV_DTYPE, device=self.device)
        if self.shift is None:
            self.shift = x.mean(dim=0)
        y = x - self.shift
        self.n += y.shape[0]
        self.sum += y.sum(dim=0)
        self.gram += y.T @ y

    def finalize(self) -> tuple[Tensor, Tensor]:
        """``(mean [d], covariance [d, d])``, both float64."""
        assert self.n > 0, "no samples accumulated"
        m = self.sum / self.n
        return self.shift + m, (self.gram - self.n * torch.outer(m, m)) / self.n


def precision_from_covariance(cov: Tensor) -> Tensor:
    """The inverse covariance, via Cholesky, which raises on a matrix that is not positive definite."""
    return torch.cholesky_inverse(torch.linalg.cholesky(cov))
