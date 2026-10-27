"""The data behind Figure 2: MVTec AD 2 accuracy against FLOPs per test image, per encoder and scorer.

FLOPs per test image are the encoder's plus the scorer's, each averaged over the 8 categories as the
accuracy is. A multiply-accumulate counts as 2 FLOPs, as in ``FlopCounterMode``.

``encoder``      one ``FlopCounterMode`` forward of the tapped call at the 640 px tile, on CPU, times
                 the category's tile count. The count does not depend on the weights, so the model
                 is built without downloading them.
``mahalanobis``  ``z^T P z`` per token per tap: ``2 d^2 + 2 d``.
``knn``          the ``[T, d] x [d, B]`` product of the 1-NN search per tap: ``2 T B d``, with ``B``
                 the bank size the run recorded for that category and tap.

The geometry and the bank sizes come from the results JSONs, the accuracy from the same files. The
output is one CSV row per (encoder, scorer), with AU-PRO@0.05 and the F1 at the EVT threshold as
the mean and sample SD over seeds.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from metacog.pipeline.backbone import TimmDinoV3Backbone
from tables_seeds import ARMS, OUT, RUN, SEEDS, dataset_mean, load

#: The results directory of each point on the curve, smallest encoder first.
RUNS = ("ad2_vits", "ad2_vitb", "ad2_vitl", "ad2")


@torch.no_grad()
def encoder_cost(backbone: str, layers: list[int], tile: int) -> tuple[int, int, int]:
    """``(parameters, FLOPs of one tile, width)`` of the call the pipeline makes."""
    from torch.utils.flop_counter import FlopCounterMode

    model = TimmDinoV3Backbone(backbone, layers, "cpu", pretrained=False)
    # FlopCounterMode does not count PyTorch's fused CPU attention kernel. The explicit q k^T and
    # attention-times-v matmuls cost the same 4 N^2 d, and are counted.
    for module in model.dino.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.zeros(1, 3, tile, tile))
    return sum(p.numel() for p in model.dino.parameters()), int(counter.get_total_flops()), model.embed_dim


def scorer_flops(arm: str, category: dict, d: int) -> float:
    """Scorer FLOPs for one test image of one category."""
    tokens = category["geometry"]["n_tokens"]
    if arm == "mahalanobis":
        return float(len(category["arms"][arm]["scorer"]["n_train_tokens"]) * tokens * (2 * d**2 + 2 * d))
    banks = category["arms"][arm]["scorer"]["bank_size"]
    return float(sum(2 * tokens * int(b) * d for b in banks.values()))


def first_results(ds: str) -> dict | None:
    for seed in SEEDS:
        path = RUN / "results" / ds / f"seed{seed}" / f"{ds}.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def main() -> None:
    rows = []
    for ds in RUNS:
        results = first_results(ds)
        if results is None:
            print(f"skip {ds}: no results")
            continue
        config, categories = results["config"], results["categories"]
        parameters, per_tile, d = encoder_cost(config["backbone"], config["layers"], config["patch_size"])
        encoder = per_tile * float(np.mean([c["geometry"]["n_tiles"] for c in categories.values()]))
        data = load(ds)
        for arm in ARMS:
            scorer = float(np.mean([scorer_flops(arm, c, d) for c in categories.values()]))
            au_pro, f1 = dataset_mean(data, arm, "au_pro@0.05"), dataset_mean(data, arm, "f1_evt")
            rows.append({
                "backbone": config["backbone"],
                "arm": arm,
                "parameters": parameters,
                "tflops_encoder": encoder / 1e12,
                "tflops_scorer": scorer / 1e12,
                "tflops_total": (encoder + scorer) / 1e12,
                "au_pro_0.05": np.mean(au_pro), "au_pro_0.05_sd": np.std(au_pro, ddof=1) if len(au_pro) > 1 else 0.0,
                "f1_evt": np.mean(f1) if f1 else float("nan"),
                "f1_evt_sd": np.std(f1, ddof=1) if len(f1) > 1 else 0.0,
                "n_seeds": len(au_pro),
            })
            print(f"  {config['backbone']:<18s} {arm:<12s} encoder {encoder / 1e12:6.2f} TFLOP  "
                  f"scorer {scorer / 1e12:6.3f} TFLOP  au_pro@0.05 {np.mean(au_pro):.4f}")
    if not rows:
        return
    OUT.mkdir(exist_ok=True)
    path = OUT / "figure2.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
