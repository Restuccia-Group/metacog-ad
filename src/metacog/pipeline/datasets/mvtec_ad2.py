"""MVTec AD 2: 8 categories, the benchmark SuperADD was built for, scored on ``test_public``.

Layout::

    <root>/<category>/train/good/*.png
    <root>/<category>/test_public/{good,bad}/*.png
    <root>/<category>/test_public/ground_truth/bad/<stem>_mask.png

The private test splits are scored by the benchmark's server and carry no labels, so they are not
read. Test file names encode lighting variants of one object (``000_regular.png``,
``000_overexposed.png``); each is a separate test image with its own mask.
"""
from __future__ import annotations

import os

from .base import AnomalyDataset, TestItem, sorted_images

CATEGORIES = (
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts",
)


class MVTecAD2(AnomalyDataset):
    name = "mvtec_ad_2"
    pro_integration_limit = 0.05
    extra_integration_limits = (0.3,)
    categories = CATEGORIES

    def train_paths(self, category: str) -> list[str]:
        return sorted_images(os.path.join(self.root, category, "train", "good"))

    def test_items(self, category: str) -> list[TestItem]:
        test_dir = os.path.join(self.root, category, "test_public")
        items: list[TestItem] = []
        for defect in ("good", "bad"):
            anomalous = defect == "bad"
            for path in sorted_images(os.path.join(test_dir, defect)):
                stem = os.path.splitext(os.path.basename(path))[0]
                mask = os.path.join(test_dir, "ground_truth", defect, f"{stem}_mask.png")
                # Upstream's dataset.py:101 lets a missing mask become an all-clean one.
                assert not anomalous or os.path.exists(mask), f"missing ground truth: {mask}"
                items.append(TestItem(path, (mask,) if anomalous else (), int(anomalous)))
        return items
