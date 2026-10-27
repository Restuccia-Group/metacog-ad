"""Image handles that decode on demand, so a training set is never held in memory at once.

Upstream's ``train.py:23`` decodes every training image to a float32 tensor up front: 25 GB for MVTec
AD 2 ``walnuts``, although ``SuperADD.train`` uses them one at a time. ``train`` indexes the list
(``model.py:191-192``) and ``predict`` reads ``x.shape`` before moving ``x`` to the device, so a
generator will not do. A :class:`LazyImage` has ``.shape``, read from the file header, and
``.to(device)``, which decodes. ``model.py`` is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class LazyImage:
    """A ``[3, H, W]`` float32 image in ``[0, 1]``, decoded only when :meth:`to` is called."""

    __slots__ = ("path", "_shape")

    def __init__(self, path: str) -> None:
        self.path = path
        with Image.open(path) as image:
            width, height = image.size
        self._shape = (3, height, width)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def to(self, device) -> torch.Tensor:
        """Decode and move. Equivalent to ``torchvision.transforms.ToTensor()(Image.open(path))``."""
        with Image.open(self.path) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
        return tensor.float().div_(255.0)


def lazy_images(paths) -> list[LazyImage]:
    return [LazyImage(path) for path in paths]
