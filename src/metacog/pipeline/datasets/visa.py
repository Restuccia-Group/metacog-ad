"""VisA: 12 categories, split by a file rather than by the directory layout.

Layout::

    <root>/split_csv/1cls.csv
    <root>/<category>/Data/Images/Normal/*.JPG      train and test normals together
    <root>/<category>/Data/Images/Anomaly/*.JPG
    <root>/<category>/Data/Masks/Anomaly/*.png

``Images/Normal/`` holds the training normals and the test normals in one directory (``candle``:
900 train and 100 test), so listing it would fit the scorer on images it is later tested on. The
split comes from ``split_csv/1cls.csv``, the one-class split the VisA authors published, which
also names each mask explicitly.

The masks are stored as 0/1 rather than 0/255, with a few stray values; ``load_gt`` thresholds at
``> 0``, so all of them read as the same boolean. The VisA literature usually resizes to 512 x 512,
so these numbers are not directly comparable with published VisA results.
"""
from __future__ import annotations

import csv
import os

from .base import AnomalyDataset, TestItem

SPLIT_CSV = ("split_csv", "1cls.csv")

CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)


class VisA(AnomalyDataset):
    name = "visa"
    # VisA has no evaluator of its own; 0.3 is the MVTec convention published VisA AU-PROs use.
    pro_integration_limit = 0.3
    extra_integration_limits = (0.05,)
    categories = CATEGORIES

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self._split: dict[str, tuple[list[str], list[TestItem]]] | None = None

    def train_paths(self, category: str) -> list[str]:
        return self._for(category)[0]

    def test_items(self, category: str) -> list[TestItem]:
        return self._for(category)[1]

    def _for(self, category: str) -> tuple[list[str], list[TestItem]]:
        assert category in CATEGORIES, f"unknown VisA category {category!r}"
        if self._split is None:
            self._split = self._read_split()
        return self._split[category]

    def _read_split(self) -> dict[str, tuple[list[str], list[TestItem]]]:
        """Parse ``1cls.csv`` into per-category train paths and test items.

        Every row is validated here, because each failure it guards against is silent: an anomaly
        in the train split, an anomalous test image without a mask (which would read as all-clean),
        or the same file in both splits.
        """
        path = os.path.join(self.root, *SPLIT_CSV)
        assert os.path.isfile(path), (
            f"VisA split file not found: {path}. Images/Normal/ holds train and test normals "
            f"together, so 1cls.csv must be present alongside the category directories."
        )
        train: dict[str, list[str]] = {category: [] for category in CATEGORIES}
        test: dict[str, list[TestItem]] = {category: [] for category in CATEGORIES}

        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"object", "split", "label", "image", "mask"}
            assert reader.fieldnames and expected <= set(reader.fieldnames), (
                f"{path} has columns {reader.fieldnames}, expected at least {sorted(expected)}"
            )
            for row_number, row in enumerate(reader, start=2):
                where = f"{path}:{row_number}"
                category, split, label = row["object"], row["split"], row["label"]
                assert category in train, f"{where} names an unknown category {category!r}"
                assert label in ("normal", "anomaly"), f"{where} has label {label!r}"
                assert split in ("train", "test"), f"{where} has split {split!r}"
                image = os.path.join(self.root, row["image"])
                assert os.path.exists(image), f"{where} references a missing image: {image}"
                if split == "train":
                    assert label == "normal", f"{where} puts an anomaly in the train split; this benchmark is one-class"
                    train[category].append(image)
                    continue
                anomalous = label == "anomaly"
                mask = os.path.join(self.root, row["mask"]) if row["mask"] else ""
                assert not anomalous or (mask and os.path.exists(mask)), (
                    f"{where} is anomalous but its mask is missing: {mask or '<empty>'}"
                )
                assert anomalous or not mask, f"{where} is normal but names a mask: {mask}"
                test[category].append(TestItem(image, (mask,) if anomalous else (), int(anomalous)))

        out: dict[str, tuple[list[str], list[TestItem]]] = {}
        for category in CATEGORIES:
            # The CSV rows are shuffled, and `train` holds out every eighth image by index, so the
            # order must not depend on how the file happens to be written.
            train_paths = sorted(train[category])
            items = sorted(test[category], key=lambda item: (item.label, item.image_path))
            assert train_paths and items, (
                f"{path} has no {'train' if not train_paths else 'test'} rows for {category!r}"
            )
            leaked = set(train_paths) & {item.image_path for item in items}
            assert not leaked, (
                f"{len(leaked)} {category} images are in both the train and test splits of {path}, "
                f"e.g. {sorted(leaked)[0]}"
            )
            out[category] = (train_paths, items)
        return out
