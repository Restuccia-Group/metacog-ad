"""The benchmark interface: normal training images, and a labelled test split.

Adapters return paths, not decoded tensors as upstream's ``MVTecAD2Dataset`` does. ``SuperADD``
does its own preprocessing on the device (``model.py:33-40``); the scale rule and the tiling need
the native size before any pixel is decoded; and ground truth belongs to the evaluation
(``pipeline/evaluation.py``), which must not inherit the NEAREST resize upstream's ``_load_mask``
applies.

Each benchmark has its own AU-PRO integration limit: 0.3 for MVTec AD, 0.05 for MVTec AD 2. The
other limit is reported next to it, read from the same PRO curve.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class TestItem:
    """One test image and its ground truth."""

    __test__ = False  # not a pytest test class

    image_path: str
    mask_paths: tuple[str, ...]   # empty for an anomaly-free image
    label: int                    # 1 anomalous, 0 normal


def sorted_images(directory: str, extensions: tuple[str, ...] = (".png",)) -> list[str]:
    """Sorted paths of the images directly inside ``directory``; empty if it does not exist."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(extensions)
    )


def native_size(path: str) -> tuple[int, int]:
    """``(height, width)`` from the image header, without decoding the pixels."""
    with Image.open(path) as image:
        width, height = image.size
    return height, width


class AnomalyDataset(ABC):
    """A one-class anomaly detection benchmark."""

    name: str = ""
    categories: tuple[str, ...] = ()
    pro_integration_limit: float = 0.3
    extra_integration_limits: tuple[float, ...] = ()

    def __init__(self, root: str) -> None:
        assert os.path.isdir(root), f"dataset root not found: {root}"
        self.root = root

    @abstractmethod
    def train_paths(self, category: str) -> list[str]:
        """Anomaly-free training images, in a stable order."""

    @abstractmethod
    def test_items(self, category: str) -> list[TestItem]:
        """Test images with labels and mask paths, in a stable order."""

    def integration_limits(self) -> tuple[float, ...]:
        """The headline limit first, then the extras."""
        return (self.pro_integration_limit, *self.extra_integration_limits)

    def native_hw(self, category: str) -> tuple[int, int]:
        """The one ``(H, W)`` every image in ``category`` shares.

        Upstream requires it but only fails at the end of training: ``np.percentile`` on the list
        of held-out maps (``model.py:218``) needs them all the same shape. So does ``compute_pro``.
        """
        paths = self.train_paths(category) + [item.image_path for item in self.test_items(category)]
        sizes = {native_size(path) for path in paths}
        assert len(sizes) == 1, (
            f"{self.name}/{category} mixes image sizes {sorted(sizes)}. SuperADD's threshold "
            f"(model.py:218) and MVTec's AU-PRO both need one size per category."
        )
        return sizes.pop()

    def config(self) -> dict:
        """Provenance for the results JSON."""
        return {
            "dataset": self.name,
            "root": self.root,
            "pro_integration_limit": self.pro_integration_limit,
            "pro_integration_limits": list(self.integration_limits()),
            "n_categories": len(self.categories),
        }
