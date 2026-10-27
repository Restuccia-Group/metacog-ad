"""The three benchmarks, by name."""
from __future__ import annotations

from .base import AnomalyDataset, TestItem
from .mvtec_ad import MVTecAD
from .mvtec_ad2 import MVTecAD2
from .visa import VisA

DATASETS: dict[str, type[AnomalyDataset]] = {cls.name: cls for cls in (MVTecAD, MVTecAD2, VisA)}


def get_dataset(name: str, root: str) -> AnomalyDataset:
    assert name in DATASETS, f"unknown dataset {name!r}; available: {sorted(DATASETS)}"
    return DATASETS[name](root)


__all__ = ["AnomalyDataset", "TestItem", "MVTecAD", "MVTecAD2", "VisA", "DATASETS", "get_dataset"]
