"""Fit and evaluate both scorer arms on one dataset.

One process per dataset, one category at a time, both arms per category. The arms are fitted one
after the other on the same data: ``np.random.seed(seed)`` is reset before each arm's ``train()``,
as upstream's ``train.py:18`` does per category, so the random brightness factor drawn per image
(``model.py:35``) is the same sequence for both.

``predict`` returns each map at ``native // evaluation_downscale``. Every metric is computed at
native resolution instead: the map is upsampled, binarized through the morphology scaled to native
size, and compared with ground truth that was never resampled, over every test image.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from industrial.model import SuperADD

from .pipeline import geometry
from .pipeline.backbone import TIMM_MODELS
from .pipeline.datasets import get_dataset
from .pipeline.evaluation import (
    assert_pro_affordable,
    au_pro_at_limits,
    image_metrics,
    load_gt,
    model_output_hw,
    morphology_for_native,
)
from .pipeline.loading import lazy_images
from .pipeline.thresholds import cross_validated_fpr, f1_for_rules, rules_from_holdout

#: The two arms, in the order every results JSON lists them.
ARMS = ("knn", "mahalanobis")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def upsample_to(anomaly_map: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of one map, the interpolation ``predict`` itself uses."""
    tensor = torch.from_numpy(np.asarray(anomaly_map, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def _au_pro_job(args):
    """Module-level so ``ProcessPoolExecutor`` can pickle it."""
    return au_pro_at_limits(*args)


def build_model(config: dict, scale: float, arm: str, device: str) -> SuperADD:
    # The kNN scorer takes its bank size and iteration count from the same keys upstream reads.
    scorer_config = (
        {k: config[k] for k in ("max_database_size", "subsampling_iterations")} if arm == "knn" else {}
    )
    return SuperADD(
        backbone=config["backbone"],
        layers=config["layers"],
        resize_factor=scale,
        patch_size=config["patch_size"],
        patch_overlap=config["patch_overlap"],
        max_database_size=config["max_database_size"],
        threshold_fraction=config["threshold_fraction"],
        subsampling_iterations=config["subsampling_iterations"],
        threshold_percentile=config["threshold_percentile"],
        threshold_factor=config["threshold_factor"],
        evaluation_downscale=config["evaluation_downscale"],
        closing_radius=config["closing_radius"],
        closing_angles=config["closing_angles"],
        closing_lower_threshold=config["closing_lower_threshold"],
        binary_erosion=config["binary_erosion"],
        brightness_augmentation=tuple(config["brightness_augmentation"]),
        device=device,
        scorer=arm,
        scorer_config=scorer_config,
    )


def evaluate_category(category: str, dataset, config: dict, args) -> dict:
    started = time.perf_counter()
    train_paths = dataset.train_paths(category)
    items = dataset.test_items(category)
    native_hw = dataset.native_hw(category)

    scale = geometry.resolve_scale(config["scale_rule"], native_hw)
    geometry.assert_geometry_is_legal(native_hw, scale, config["patch_size"])
    geom = geometry.summary(native_hw, scale, config["patch_size"], config["patch_overlap"])
    output_hw = model_output_hw(native_hw, config["evaluation_downscale"])
    print(
        f"\n{'=' * 78}\n{dataset.name} / {category}\n{'=' * 78}\n"
        f"  native {native_hw[0]}x{native_hw[1]}  scale {scale:.6f}  -> "
        f"{geom['downscaled_hw'][0]}x{geom['downscaled_hw'][1]}  "
        f"{geom['n_tiles']} tiles {geom['tiles_per_axis']}  grid {geom['grid_hw']}  "
        f"({geom['n_tokens']} tokens, {geom['native_px_per_token'][0]:.1f} native px/token, "
        f"dropped {geom['dropped_px']} px)\n"
        f"  {len(train_paths)} train / {len(items)} test  |  model output {output_hw}"
    )

    train_images = lazy_images(train_paths)
    labels = np.array([item.label for item in items], dtype=np.uint8)
    masks = [load_gt(item.mask_paths, native_hw) for item in items]
    limits = dataset.integration_limits()
    assert_pro_affordable(len(items), native_hw, args.pro_memory_budget_gb * 10**9)
    postprocess = morphology_for_native(config)
    cuda = args.device.startswith("cuda")

    arms_out: dict[str, dict] = {}
    pro_jobs: dict[str, tuple] = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        # Reset before each arm, so both draw the same brightness factors.
        np.random.seed(args.seed)
        model = build_model(config, scale, arm, args.device)

        # `train` scores the held-out normal images through `predict` to fit its threshold, then
        # drops the maps (model.py:213-218). Every threshold rule is derived from them.
        holdout: list[np.ndarray] = []
        model.map_sink = holdout.append
        fit_started = time.perf_counter()
        model.train(train_images)
        fit_seconds = time.perf_counter() - fit_started
        model.map_sink = None

        score_started = time.perf_counter()
        maps = [
            upsample_to(model.predict(lazy_images([item.image_path])[0])[0], native_hw)
            for item in items
        ]
        score_seconds = time.perf_counter() - score_started

        summary = model.scorer.summary()
        threshold = float(model.threshold)
        gpu_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        del model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

        metrics = image_metrics(np.array([m.max() for m in maps]), labels)
        holdout = [upsample_to(m, native_hw) for m in holdout]
        thresholds = rules_from_holdout(holdout)
        fpr_cv = cross_validated_fpr(holdout, args.seed)
        del holdout
        rule_f1 = f1_for_rules(maps, masks, thresholds, postprocess, args.f1_workers)
        metrics.update({f"f1_{rule}": value for rule, value in rule_f1.items()})
        # AU-PRO runs after both arms, so the two can share a process pool.
        pro_jobs[arm] = (maps, masks, limits)

        arms_out[arm] = {
            "scorer": summary,
            "threshold": threshold,
            "thresholds": thresholds,
            "evt_fpr_cv": fpr_cv,
            "metrics": metrics,
            "seconds": {
                "fit": fit_seconds,
                "score": score_seconds,
                "per_test_image": score_seconds / max(1, len(items)),
                "total": time.perf_counter() - arm_started,
            },
            "memory": {
                "gpu_peak_gb": gpu_peak_gb,
                "process_peak_rss_gb": peak_rss_gb(),
                "model_bytes": summary["model_bytes"],
            },
        }
        print(
            f"  {arm:12s} fit {fit_seconds:6.0f}s  score {score_seconds:6.0f}s "
            f"({score_seconds / max(1, len(items)) * 1000:5.0f} ms/img)  "
            f"threshold {threshold:.4f}  model {summary['model_bytes'] / 1e6:7.1f} MB  "
            f"gpu {gpu_peak_gb:.1f} GB"
        )

    pro_started = time.perf_counter()
    jobs = [pro_jobs[arm] for arm in ARMS]
    if args.pro_workers > 1:
        # Each worker gets a pickled copy of its maps and masks (3.9 GB for MVTec AD 2 `fabric`),
        # which is why this is off by default.
        with ProcessPoolExecutor(max_workers=min(args.pro_workers, len(jobs))) as pool:
            results = list(pool.map(_au_pro_job, jobs))
    else:
        results = [_au_pro_job(job) for job in jobs]
    for arm, au_pros in zip(ARMS, results):
        headline = au_pros[dataset.pro_integration_limit]
        arms_out[arm]["metrics"].update({f"au_pro@{limit:g}": v for limit, v in au_pros.items()})
        arms_out[arm]["metrics"]["au_pro"] = headline
        arms_out[arm]["seconds"]["au_pro"] = (time.perf_counter() - pro_started) / len(jobs)
        print(f"  {arm:12s} au_pro@{dataset.pro_integration_limit:g} {headline:.4f}  "
              f"f1 evt {arms_out[arm]['metrics']['f1_evt'] * 100:5.2f}")
    del pro_jobs, jobs
    gc.collect()

    return {
        "counts": {
            "n_train": len(train_paths),
            "n_test": len(items),
            "n_anomalous": int(labels.sum()),
            "n_good": int((labels == 0).sum()),
        },
        "geometry": geom,
        "frames": {"native": list(native_hw), "model_output": list(output_hw)},
        "arms": arms_out,
        "seconds": time.perf_counter() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, help="directory of the dataset the config names")
    parser.add_argument("--category", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="upstream's, train.py:18")
    parser.add_argument("--pro-workers", type=int, default=1,
                        help="2 computes the two arms' AU-PRO in parallel, at the cost of a pickled "
                             "copy of each arm's maps")
    parser.add_argument("--pro-memory-budget-gb", type=float, default=120.0)
    parser.add_argument("--f1-workers", type=int, default=os.cpu_count() or 1,
                        help="processes for the f1max gain sweep, the longest CPU step of a run")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as handle:
        config = json.load(handle)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False   # PyTorch's default, and what upstream runs with

    dataset = get_dataset(config["dataset"]["name"], args.data_root)
    categories = list(dataset.categories) if args.category == "all" else args.category.split(",")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Categories already in the output are not recomputed, so a restart after a crash resumes
    # where the run stopped. Failed categories are retried.
    per_category: dict[str, dict] = {}
    if os.path.exists(args.output):
        try:
            with open(args.output) as handle:
                previous = json.load(handle).get("categories", {})
            per_category = {k: v for k, v in previous.items() if "arms" in v}
            if per_category:
                print(f"resuming: {len(per_category)} categories already done "
                      f"({', '.join(sorted(per_category))})")
        except (json.JSONDecodeError, OSError):
            print(f"note: {args.output} unreadable, starting fresh")

    for category in categories:
        if category in per_category:
            continue
        try:
            per_category[category] = evaluate_category(category, dataset, config, args)
        except Exception as error:  # keep an unattended run going past one bad category
            traceback.print_exc()
            print(f"  !! {category} FAILED: {type(error).__name__}: {error}", flush=True)
            per_category[category] = {"error": f"{type(error).__name__}: {error}"}
        write_results(args, config, dataset, per_category)
    done = sum("arms" in v for v in per_category.values())
    print(f"\n{done}/{len(categories)} categories complete\nwrote {args.output}")
    # A non-zero exit tells the driver to retry.
    raise SystemExit(0 if done == len(categories) else 1)


def write_results(args, config, dataset, per_category):
    """Rewritten after every category, so an interrupted run keeps what it finished."""
    results = {
        "config": {
            **dataset.config(),
            **{k: v for k, v in config.items() if k != "dataset"},
            "arms": list(ARMS),
            "backbone_timm": TIMM_MODELS[config["backbone"]],
            "seed": args.seed,
            "device": args.device,
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "categories": per_category,
    }
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
