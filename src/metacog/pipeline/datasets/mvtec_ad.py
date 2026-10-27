"""MVTec AD: 15 categories.

Layout::

    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect>/*.png          ("good" plus one directory per defect type)
    <root>/<category>/ground_truth/<defect>/<stem>_mask.png

Every category is square, so the scale rule puts each one on a single 640 x 640 tile. There is no
validation split, so the threshold comes from SuperADD's held-out eighth of the training set
(``model.py:191``), as it does on the other two benchmarks.
"""
from __future__ import annotations

import os

from .base import AnomalyDataset, TestItem, sorted_images

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


class MVTecAD(AnomalyDataset):
    name = "mvtec_ad"
    pro_integration_limit = 0.3
    extra_integration_limits = (0.05,)
    categories = CATEGORIES

    def train_paths(self, category: str) -> list[str]:
        return sorted_images(os.path.join(self.root, category, "train", "good"))

    def test_items(self, category: str) -> list[TestItem]:
        test_dir = os.path.join(self.root, category, "test")
        items: list[TestItem] = []
        for defect in sorted(os.listdir(test_dir)):
            anomalous = defect != "good"
            for path in sorted_images(os.path.join(test_dir, defect)):
                stem = os.path.splitext(os.path.basename(path))[0]
                mask = os.path.join(self.root, category, "ground_truth", defect, f"{stem}_mask.png")
                assert not anomalous or os.path.exists(mask), f"missing ground truth: {mask}"
                items.append(TestItem(path, (mask,) if anomalous else (), int(anomalous)))
        return items
