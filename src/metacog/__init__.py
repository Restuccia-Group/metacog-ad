"""A frozen encoder (the object level) and a swappable scorer (the meta level).

``metacog.pipeline`` holds everything both arms share: the encoder, the tiling, the datasets, the
threshold rules and the metrics. ``metacog.scorers`` holds the one thing that differs:
``knn.py`` (the memory bank) and ``mahalanobis.py`` (one Gaussian per layer), behind the interface
in ``base.py``. ``metacog.run`` runs the comparison.

The pipeline itself is upstream SuperADD, in ``tracks/industrial/``. It calls into this package
from ``industrial.model.SuperADD``, which builds a scorer and calls ``partial_fit`` and ``distance``
on it. ``docs/UPSTREAM_DIFF.md`` lists every change made to upstream to allow that.

A reference like ``model.py:218`` is a line of **upstream's** ``model.py`` at commit ``44cf251``,
not of the modified copy in ``tracks/``.
"""
