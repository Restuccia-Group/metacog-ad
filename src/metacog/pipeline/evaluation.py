"""The metrics the paper reports (AU-PRO, image AUROC and pooled segmentation F1), and the ground truth.

Everything is computed at native resolution, against ground truth that was never resampled, over
every test image including the defect-free ones. SuperADD's ``test.py`` evaluates differently in
two ways, and both change the numbers:

- It resizes the ground truth to the map with ``cv2.resize(gt, ..., INTER_LINEAR) > 0``
  (``test.py:59``). At a factor of 4 that both dilates and drops regions: on MVTec AD 2 the
  connected-component count moves by -11% to +57% (``fruit_jelly`` 228 -> 357). AU-PRO weights
  every region equally, so it is sensitive to exactly this.
- It skips the anomaly-free test images (``test.py:41``), 34% of MVTec AD 2's public test split.
  They are the images where a detector can only produce false positives.

Here every map is upsampled to native instead, :func:`load_gt` refuses a mask of the wrong size,
and every test image is scored.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from ..vendor.mvtec_pro import compute_pro, trapezoid

# Peak bytes per pixel inside `compute_pro`: the input list, np.array's copy, the sort indices and
# their int64 argsort temporary, the float64 `pro_changes` and the mask copies all coexist. Measured
# on MVTec AD 2 `can` (162 images of 1024x2232, 370 Mpx), which peaked at 33 GB RSS.
PRO_BYTES_PER_PIXEL = 89


def assert_pro_affordable(n_images: int, hw: tuple[int, int], budget_bytes: int) -> None:
    """Fail before the encoder runs, rather than being OOM-killed at the AU-PRO step hours later."""
    pixels = n_images * hw[0] * hw[1]
    needed = pixels * PRO_BYTES_PER_PIXEL
    # `compute_pro` casts its sort indices to uint32.
    assert pixels < 2**32, (
        f"AU-PRO over {n_images} images of {hw[0]}x{hw[1]} is {pixels} pixels, past the uint32 "
        f"limit the vendored compute_pro asserts on"
    )
    assert needed <= budget_bytes, (
        f"AU-PRO over {n_images} images of {hw[0]}x{hw[1]} needs ~{needed / 1e9:.1f} GB, over the "
        f"{budget_bytes / 1e9:.1f} GB budget. Lower --pro-workers, or evaluate this category alone."
    )


def au_pro_at_limits(anomaly_maps: list[np.ndarray], masks: list[np.ndarray], limits) -> dict[float, float]:
    """AU-PRO at each FPR limit, normalized by the limit so a perfect detector scores 1.0.

    Every limit is read off one PRO curve. The curve is the expensive part, and sharing it means
    AU-PRO@0.05 and AU-PRO@0.3 are two readings of the same measurement.
    """
    assert len(anomaly_maps) == len(masks), "map/mask counts disagree"
    assert any(mask.any() for mask in masks), "no anomalous pixels: AU-PRO is undefined"
    limits = list(limits)
    assert limits and all(0 < limit <= 1 for limit in limits), f"bad integration limits {limits}"

    fprs, pros = compute_pro(
        anomaly_maps=[np.asarray(m, dtype=np.float32) for m in anomaly_maps],
        ground_truth_maps=[np.asarray(m, dtype=np.uint8) for m in masks],
    )
    return {limit: float(trapezoid(fprs, pros, x_max=limit) / limit) for limit in limits}


def image_metrics(image_scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Image AUROC, scoring each image by the max of its anomaly map."""
    if labels.max() == 0 or labels.min() == 1:
        return {"image_auroc": float("nan")}
    return {"image_auroc": float(roc_auc_score(labels, image_scores))}


def morphology_for_native(config: dict):
    """SuperADD's post-processing, with its lengths scaled from the model's output to native size.

    Upstream tuned ``closing_radius`` and ``binary_erosion`` in pixels of the map ``predict``
    returns, which is ``native // evaluation_downscale``. Scaling both by ``evaluation_downscale``
    keeps the same operation in image terms at native resolution. ``closing_angles`` is a count and
    ``closing_lower_threshold`` a fraction, so neither scales.
    """
    from industrial.model import PostProcessing

    factor = int(config["evaluation_downscale"])
    return PostProcessing(
        int(config["closing_radius"]) * factor,
        int(config["closing_angles"]),
        float(config["closing_lower_threshold"]),
        int(config["binary_erosion"]) * factor,
    )


def f1_at_threshold(anomaly_maps, masks, threshold: float, postprocess=None) -> float:
    """Pixel F1 pooled over every image, at one threshold.

    With ``postprocess=None`` the map is thresholded directly. Otherwise ``postprocess`` is an
    ``industrial.model.PostProcessing``, and the map goes through upstream's closing, fill and
    erosion (``model.py:150-154``) before it is compared.
    """
    tp = fp = fn = 0
    for anomaly_map, mask in zip(anomaly_maps, masks):
        if postprocess is None:
            predicted = np.asarray(anomaly_map) > threshold
        else:
            predicted = np.asarray(postprocess(np.asarray(anomaly_map), threshold)) > 0
        truth = np.asarray(mask).astype(bool)
        hits = int(np.logical_and(predicted, truth).sum())
        tp += hits
        fp += int(predicted.sum()) - hits
        fn += int(truth.sum()) - hits
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def model_output_hw(native_hw: tuple[int, int], evaluation_downscale: int) -> tuple[int, int]:
    """The size ``predict`` returns its map at: ``native // evaluation_downscale`` (``model.py:227``)."""
    return (native_hw[0] // evaluation_downscale, native_hw[1] // evaluation_downscale)


def load_gt(mask_paths: tuple[str, ...], hw: tuple[int, int]) -> np.ndarray:
    """The union of ``mask_paths`` as a boolean array. No paths means an anomaly-free image."""
    out = np.zeros(hw, dtype=bool)
    for path in mask_paths:
        raw = np.asarray(Image.open(path).convert("L"))
        assert raw.shape == tuple(hw), (
            f"ground truth must already be at the requested size: {raw.shape} != {hw}"
        )
        out |= raw > 0
    return out
