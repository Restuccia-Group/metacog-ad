"""Threshold rules derived from held-out normal pixels, and the F1 each one scores.

SuperADD fixes one operating point, ``np.percentile(holdout_maps, 95) * 1.421`` (``model.py:218``):
the percentile comes from its held-out normals, the gain is a constant. The rules here are all
derived from the same held-out normals, pooled at native resolution, and none of them reads a test
mask except :data:`ORACLE`:

``q95``       the 95th percentile, i.e. a 5% per-pixel false-positive rate
``q99``       the 99th percentile
``mu3sigma``  ``mean + 3 * sd``, the rule MVTec AD 2 prescribes; it assumes a Gaussian tail
``evt``       a generalized Pareto tail fitted above ``q99``, read at a normal FPR of :data:`ALPHA`
``inherited`` upstream's ``1.421 * q95``
``f1max``     the best F1 over a grid of gains on ``q95``, chosen **with the test masks**. It is a
              ceiling on what re-tuning upstream's constant can buy, not a deployable rule

Thresholds are fitted on the upsampled map rather than on the token grid, because they are applied
there, and bilinear interpolation shrinks the extreme tail.

:func:`cross_validated_fpr` is how the paper chose ``k``: it measures the false-positive rate the
EVT threshold actually delivers on held-out normal images, for ``k`` = 3, 4 and 5.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np

from .evaluation import f1_at_threshold

#: The normal-pixel false-positive rate the EVT threshold is set to: the paper's
#: ``alpha_k = 1 - Phi(k)`` at ``k = 4`` (3.17e-5), rounded to 3e-5 as in every reported run.
ALPHA = 3e-5

#: Upstream's gain on ``q95``.
INHERITED = 1.421

#: The percentile the GPD is fitted above. It leaves 1% of the holdout as excesses (62k-116k
#: pixels per category), enough for a two-parameter fit and far enough out for the GPD limit.
POT_PERCENTILE = 99.0

#: The oracle column. It reads the test masks.
ORACLE = "f1max"

#: Every rule, in the order a table reads them.
RULES = ("q95", "q99", "mu3sigma", "evt", "inherited", ORACLE)

#: The gains the oracle sweeps on ``q95``: a 0.02 grid, which is finer than the F1 peak is wide,
#: plus the round values and upstream's 1.421 so the sweep can never score below upstream.
GAINS = tuple(sorted(
    set(np.round(np.arange(0.90, 2.101, 0.02), 3).tolist())
    | {1.0, 1.1, 1.2, 1.25, 1.3, 1.4, INHERITED, 1.5, 1.6, 1.75, 2.0}
))


def gpd_quantile(alpha: float, u: float, xi: float, sigma: float, zeta_u: float) -> float:
    """The ``1 - alpha`` quantile of the parent distribution, from a POT fit above ``u``.

    ``u + (sigma / xi) * ((alpha / zeta_u)^-xi - 1)``, or ``u - sigma * log(alpha / zeta_u)`` in the
    ``xi -> 0`` limit. ``zeta_u`` is the fraction of samples above ``u``.
    """
    ratio = alpha / zeta_u
    if abs(xi) < 1e-8:
        return float(u - sigma * np.log(ratio))
    return float(u + (sigma / xi) * (ratio ** (-xi) - 1.0))


def fit_tail(pooled: np.ndarray) -> tuple[float, float, float]:
    """``(u, xi, sigma)``: a generalized Pareto fitted to the excesses above ``u = q99``.

    The location is pinned at zero, which is the POT parametrization: a free location would move
    the threshold the fit is conditioned on.
    """
    from scipy.stats import genpareto

    u = float(np.percentile(pooled, POT_PERCENTILE))
    xi, _, sigma = genpareto.fit(pooled[pooled > u] - u, floc=0.0)
    return u, float(xi), float(sigma)


def evt_threshold(tail: tuple[float, float, float], alpha: float = ALPHA) -> float:
    """The ``1 - alpha`` quantile of a :func:`fit_tail` fit."""
    return gpd_quantile(alpha, *tail, 1.0 - POT_PERCENTILE / 100.0)


def rules_from_holdout(holdout_maps: list[np.ndarray]) -> dict[str, float]:
    """Every derivable threshold, from held-out normal maps already at the evaluation resolution."""
    assert holdout_maps, "no held-out normal maps: cannot derive a threshold"
    pooled = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in holdout_maps])
    q95 = float(np.percentile(pooled, 95))
    return {
        "q95": q95,
        "q99": float(np.percentile(pooled, POT_PERCENTILE)),
        "mu3sigma": float(pooled.mean() + 3.0 * pooled.std(ddof=1)),
        "evt": evt_threshold(fit_tail(pooled)),
        "inherited": INHERITED * q95,
    }


#: The ``k`` values of ``alpha_k = 1 - Phi(k)`` the cross-validation compares, and its fold count.
CV_KS = (3, 4, 5)
CV_FOLDS = 5


def alpha_k(k: float) -> float:
    """``1 - Phi(k)``: the per-pixel false-positive rate of a ``k``-sigma rule on a Gaussian."""
    from scipy.stats import norm

    return float(norm.sf(k))


def cross_validated_fpr(holdout_maps: list[np.ndarray], seed: int, ks=CV_KS, folds: int = CV_FOLDS) -> dict:
    """The per-pixel FPR the EVT threshold delivers on normal images it was not fitted on.

    The held-out normal images are shuffled with ``seed`` and split into ``folds`` folds, image by
    image rather than pixel by pixel: neighbouring pixels of one map are interpolants of each other,
    so a pixel split would leak the tail it is meant to test. Each fold is thresholded by an EVT
    fit on the other folds at ``alpha_k``. Returns ``{k: {"alpha", "exceed", "pixels"}}``, the
    exceedances and pixels summed over the folds, so the FPR pools over folds and categories alike.
    """
    assert len(holdout_maps) >= folds, (
        f"{len(holdout_maps)} held-out images cannot be split into {folds} folds"
    )
    flat = [np.asarray(m, dtype=np.float32).ravel() for m in holdout_maps]
    order = np.random.default_rng(seed).permutation(len(flat))
    out = {str(k): {"alpha": alpha_k(k), "exceed": 0, "pixels": 0} for k in ks}
    for fold in range(folds):
        held = set(order[fold::folds].tolist())
        tail = fit_tail(np.concatenate([flat[i] for i in order if i not in held]))
        test = np.concatenate([flat[i] for i in sorted(held)])
        for k in ks:
            record = out[str(k)]
            record["exceed"] += int((test > evt_threshold(tail, record["alpha"])).sum())
            record["pixels"] += int(test.size)
    return out


# Set before the pool forks, so the workers inherit the maps by copy-on-write instead of receiving
# a pickled copy with every task.
_MAPS: list = []
_MASKS: list = []
_POSTPROCESS = None


def _f1_at(threshold: float) -> float:
    return f1_at_threshold(_MAPS, _MASKS, threshold, _POSTPROCESS)


def f1_for_rules(maps, masks, thresholds: dict, postprocess, jobs: int) -> dict[str, float]:
    """``{rule: F1}`` for every threshold in ``thresholds``, plus :data:`ORACLE`.

    The oracle sweep is one full morphology pass per gain and dominates the cost, so it runs on
    ``jobs`` processes.
    """
    global _MAPS, _MASKS, _POSTPROCESS
    _MAPS, _MASKS, _POSTPROCESS = maps, masks, postprocess
    out = {rule: _f1_at(value) for rule, value in thresholds.items()}
    sweep = [thresholds["q95"] * gain for gain in GAINS]
    if jobs > 1:
        with Pool(processes=jobs) as pool:
            f1s = pool.map(_f1_at, sweep, chunksize=1)
    else:
        f1s = [_f1_at(t) for t in sweep]
    out[ORACLE] = max(f1s)
    return out
