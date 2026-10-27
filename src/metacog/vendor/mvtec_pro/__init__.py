"""MVTec's official AU-PRO implementation, vendored unmodified.

``pro_curve_util.py`` and ``generic_util.py`` are byte-for-byte copies of the MVTec AD evaluation
scripts (BSD-3-Clause, Copyright 2022 MVTec Software GmbH; see
``third_party_licenses/MVTecEval_BSD3.txt``). They are not edited, so the AU-PRO here is the
official one, per-region weighting and FPR denominator included.

``pro_curve_util`` imports ``label`` from ``scipy.ndimage.measurements``, which SciPy has
deprecated; the alias below keeps that import working once the module is gone. ``generic_util``
imports ``tifffile`` for a helper nothing here calls, so ``tifffile`` is a dependency.
"""
from __future__ import annotations

import sys

import scipy.ndimage

# Installed before the vendored modules are imported.
if "scipy.ndimage.measurements" not in sys.modules:  # pragma: no cover - trivial
    try:
        import scipy.ndimage.measurements  # noqa: F401
    except ImportError:
        sys.modules["scipy.ndimage.measurements"] = scipy.ndimage

from .generic_util import trapezoid  # noqa: E402
from .pro_curve_util import compute_pro  # noqa: E402

__all__ = ["compute_pro", "trapezoid"]
