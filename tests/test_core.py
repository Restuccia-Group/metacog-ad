"""Offline checks: no GPU, no dataset, no network.

They pin what would otherwise fail silently: the upstream files and geometry, the one-pass
covariance, the scorer interface, the metrics, the dataset splits and the threshold rules.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
import torch

from metacog.pipeline import geometry
from metacog.pipeline.evaluation import au_pro_at_limits, model_output_hw
from metacog.scorers import KnnScorer, MahalanobisScorer, build_scorer
from metacog.scorers.covariance import StreamingCovariance, precision_from_covariance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- upstream integrity ---------------------------------------------------------------------

# The two upstream files this repository uses unmodified.
UPSTREAM_SHA256 = {
    "nearest_neighbor.py": "a0daeb8df649c255bb014baf3a99d62255540ea05c702de33d08d507520d87f5",
    "post_process.py": "3150b5eb7fda2c8943bf627c69831c39342cd9962c9eeb86d8b7c2dd90cd5585",
}


@pytest.mark.parametrize("filename,expected", sorted(UPSTREAM_SHA256.items()))
def test_upstream_files_are_byte_identical(filename, expected):
    path = os.path.join(REPO, "tracks/industrial/src/industrial", filename)
    with open(path, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == expected, (
            f"{filename} has been modified. It must stay byte-identical to upstream."
        )


# --- geometry -------------------------------------------------------------------------------

# (native size, token grid, tile count) under SuperADD's int(side * scale) // 16. Rounding to the
# patch grid instead would give fabric 80x96.
AD2_GEOMETRY = {
    "can": ((1024, 2232), (40, 87), 3),
    "fabric": ((2048, 2448), (80, 95), 12),
    "fruit_jelly": ((1520, 2100), (59, 82), 6),
    "rice": ((2048, 2448), (80, 95), 12),
    "sheet_metal": ((1056, 4224), (41, 165), 14),
    "vial": ((1900, 1400), (74, 54), 6),
    "wallplugs": ((2048, 2448), (80, 95), 12),
    "walnuts": ((2048, 2448), (80, 95), 12),
}

RULE = {"scale": 0.625, "min_side": 640}
MVTEC_AD_SIDES = (700, 800, 840, 900, 1000, 1024)
# VisA native sizes. Five have a short side under 1024 px, where 0.625 is illegal.
VISA_HW = {
    "candle": (1168, 1284), "capsules": (1000, 1500), "cashew": (1176, 1274),
    "chewinggum": (1118, 1342), "fryum": (1000, 1500), "macaroni1": (1000, 1500),
    "macaroni2": (1000, 1500), "pcb1": (1070, 1404), "pcb2": (1070, 1404), "pcb3": (960, 1562),
    "pcb4": (1104, 1358), "pipe_fryum": (1154, 1300),
}


@pytest.mark.parametrize("category", sorted(AD2_GEOMETRY))
def test_ad2_geometry_is_upstreams(category):
    """On MVTec AD 2 the scale rule is exactly upstream's 0.625, and so is the layout."""
    native_hw, grid, n_tiles = AD2_GEOMETRY[category]
    scale = geometry.resolve_scale(RULE, native_hw)
    assert scale == 0.625
    summary = geometry.summary(native_hw, scale)
    assert tuple(summary["grid_hw"]) == grid
    assert summary["n_tiles"] == n_tiles


@pytest.mark.parametrize("side", MVTEC_AD_SIDES)
def test_mvtec_ad_lands_on_exactly_one_640_tile(side):
    scale = geometry.resolve_scale(RULE, (side, side))
    assert geometry.downscaled_size((side, side), scale) == (640, 640)
    summary = geometry.summary((side, side), scale)
    assert summary["n_tiles"] == 1 and tuple(summary["grid_hw"]) == (40, 40)


@pytest.mark.parametrize("category", sorted(VISA_HW))
def test_visa_is_legal_and_no_coarser_than_mvtec_ad_2(category):
    """MVTec AD 2 is encoded at 25.6 native px per token. VisA stays within 26.2: the extra is the
    up to 15 px that ``int(side * scale) // 16`` drops, which gives MVTec AD 2 25.8 on two categories."""
    native_hw = VISA_HW[category]
    scale = geometry.resolve_scale(RULE, native_hw)
    geometry.assert_geometry_is_legal(native_hw, scale)
    assert scale >= 0.625
    assert native_hw[0] / geometry.summary(native_hw, scale)["grid_hw"][0] <= 26.2


def test_can_sits_exactly_on_the_tile_boundary():
    """``can`` downscales from 1024 to exactly 640 px, so it is legal by zero margin."""
    geometry.assert_geometry_is_legal((1024, 2232), 0.625)
    with pytest.raises(AssertionError, match="min_side"):
        geometry.assert_geometry_is_legal((1023, 2232), 0.625)


def test_a_bare_0625_is_rejected_where_upstream_would_crash():
    sizes = [(s, s) for s in MVTEC_AD_SIDES if int(s * 0.625) < 640]
    sizes += [hw for hw in VISA_HW.values() if int(min(hw) * 0.625) < 640]
    assert len(sizes) == 5 + 5
    for native_hw in sizes:
        with pytest.raises(AssertionError, match=r"min_side to 640"):
            geometry.assert_geometry_is_legal(native_hw, 0.625)


def test_model_output_hw_is_floor_division_on_native():
    assert model_output_hw((1520, 2100), 4) == (380, 525)
    assert model_output_hw((1900, 1400), 4) == (475, 350)
    assert model_output_hw((1023, 1025), 4) == (255, 256)


# --- covariance -----------------------------------------------------------------------------

@pytest.mark.parametrize("n,d,offset", [(3000, 32, 0.0), (3000, 64, 40.0), (5000, 128, 12.0)])
def test_one_pass_matches_sklearn(n, d, offset):
    """The streaming fit against sklearn. ``offset`` exercises the shift."""
    from sklearn.covariance import empirical_covariance

    torch.manual_seed(0)
    x = torch.randn(n, d, dtype=torch.float64) @ (torch.randn(d, d, dtype=torch.float64) / d**0.5)
    x = x + offset

    accumulator = StreamingCovariance(d, torch.device("cpu"))
    for start in range(0, n, 397):          # ragged blocks
        accumulator.update(x[start : start + 397])
    mean, cov = accumulator.finalize()

    assert torch.allclose(mean, x.mean(0), rtol=1e-10)
    assert torch.allclose(cov, torch.as_tensor(empirical_covariance(x.numpy())), rtol=1e-9, atol=1e-12)


def test_one_pass_does_not_depend_on_the_block_size():
    torch.manual_seed(1)
    x = torch.randn(2000, 48, dtype=torch.float64) * 3 + 7
    whole = StreamingCovariance(48, torch.device("cpu"))
    whole.update(x)
    blocks = StreamingCovariance(48, torch.device("cpu"))
    for start in range(0, 2000, 101):
        blocks.update(x[start : start + 101])
    for left, right in zip(whole.finalize(), blocks.finalize()):
        assert torch.allclose(left, right, rtol=1e-10)


def test_the_estimator_is_the_sample_covariance_divided_by_n():
    torch.manual_seed(4)
    x = torch.randn(3000, 16, dtype=torch.float64) * 2 + 5
    accumulator = StreamingCovariance(16, torch.device("cpu"))
    accumulator.update(x)
    _, cov = accumulator.finalize()
    centered = x - x.mean(0)
    assert torch.allclose(cov, centered.T @ centered / len(x), rtol=1e-9, atol=1e-12)


def test_precision_rejects_a_non_positive_definite_covariance():
    with pytest.raises(Exception):
        precision_from_covariance(torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64))


# --- scorers --------------------------------------------------------------------------------

def _blocks(n_images=6, tokens=200, d=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(tokens, d, generator=generator).numpy() for _ in range(n_images)]


@pytest.mark.parametrize("kind", ["knn", "mahalanobis"])
def test_scorer_interface(kind):
    layers = [1, 2]
    config = {"max_database_size": 300, "subsampling_iterations": 3} if kind == "knn" else {}
    scorer = build_scorer(kind, layers, "cpu", **config)
    for block in _blocks():
        for layer in layers:
            scorer.partial_fit(layer, block)
    scorer.finalize()

    query = torch.randn(50, 32)
    for layer in layers:
        out = scorer.distance(layer, query)
        assert out.shape == (50,) and out.device == query.device
        assert torch.isfinite(out).all() and (out >= 0).all()
    assert scorer.model_bytes() > 0
    assert scorer.summary()["kind"] == kind


def test_the_mahalanobis_model_size_is_a_mean_and_a_precision_per_layer():
    """The paper's 52.47 MB is four layers of a float64 [1280] mean and [1280, 1280] precision."""
    scorer = MahalanobisScorer([1, 2], "cpu")
    for block in _blocks(d=16):
        scorer.partial_fit(1, block)
        scorer.partial_fit(2, block)
    scorer.finalize()
    assert scorer.model_bytes() == 2 * (16 + 16 * 16) * 8
    assert 4 * (1280 + 1280 * 1280) * 8 / 1e6 == pytest.approx(52.47, abs=0.005)


def test_mahalanobis_map_is_the_square_root_and_dimension_normalized():
    """``sqrt(d^2 / c)``, so the 1.421 gain means for this arm what it means for kNN."""
    d = 16
    scorer = MahalanobisScorer([1], "cpu")
    for block in _blocks(n_images=40, tokens=200, d=d):
        scorer.partial_fit(1, block)
    scorer.finalize()

    query = torch.randn(100, d)
    diff = query.double() - scorer._mean[1]
    expected = torch.sqrt(torch.einsum("nd,de,ne->n", diff, scorer._precision[1], diff) / d).float()
    assert torch.allclose(scorer.distance(1, query), expected, rtol=1e-5)


def test_knn_scores_with_k_equals_one():
    """Upstream's scoring rule. The paper's k = 100 is the bank subsampling parameter."""
    import inspect

    from metacog.scorers.knn import SCORING_K, SUBSAMPLING_K

    assert SCORING_K == 1 and SUBSAMPLING_K == 100
    assert KnnScorer([1], "cpu").summary()["k"] == 1
    assert "k" not in inspect.signature(KnnScorer.__init__).parameters


def test_both_arms_are_registered_scorers():
    from metacog.run import ARMS
    from metacog.scorers import SCORERS

    assert set(ARMS) == set(SCORERS)


# --- AU-PRO ---------------------------------------------------------------------------------

def test_load_gt_refuses_a_mask_that_is_not_already_native(tmp_path):
    from PIL import Image

    from metacog.pipeline.evaluation import load_gt

    path = os.path.join(str(tmp_path), "mask.png")
    Image.new("L", (40, 30)).save(path)
    assert load_gt((path,), (30, 40)).shape == (30, 40)
    with pytest.raises(AssertionError, match="must already be at the requested size"):
        load_gt((path,), (15, 20))


def test_au_pro_is_perfect_for_a_perfect_map_and_low_for_a_flat_one():
    masks = [np.zeros((64, 64), dtype=np.uint8) for _ in range(4)]
    masks[0][10:20, 10:20] = 1
    masks[1][40:50, 40:50] = 1
    perfect = [m.astype(np.float32) for m in masks]
    assert au_pro_at_limits(perfect, masks, [0.05])[0.05] == pytest.approx(1.0, abs=1e-6)
    flat = [np.zeros((64, 64), dtype=np.float32) for _ in masks]
    assert au_pro_at_limits(flat, masks, [0.3])[0.3] < 0.6


def test_au_pro_chance_is_half_the_integration_limit():
    """A random map traces PRO = FPR, so chance is ``limit / 2`` (0.15 at 0.3), not 0.5."""
    rng = np.random.default_rng(0)
    masks = [np.zeros((64, 64), dtype=np.uint8) for _ in range(24)]
    for i in range(12):
        masks[i][10:20, 10:20] = 1
    maps = [rng.random((64, 64)).astype(np.float32) for _ in masks]
    areas = au_pro_at_limits(maps, masks, [0.3, 0.05])
    assert areas[0.3] == pytest.approx(0.15, abs=0.03)
    assert areas[0.05] == pytest.approx(0.025, abs=0.02)


def test_au_pro_depends_only_on_the_ranking():
    """The square root and ``/ c`` in the Mahalanobis map are monotone, so they cannot move AU-PRO."""
    rng = np.random.default_rng(7)
    maps = [rng.random((12, 12)) for _ in range(5)]
    masks = [np.zeros((12, 12), dtype=np.uint8) for _ in range(5)]
    masks[1][3:6, 3:6] = 1
    masks[3][7:9, 2:5] = 1
    base = au_pro_at_limits(maps, masks, [0.05, 0.3])
    for transformed in ([m * 3.7 for m in maps], [np.sqrt(m) / 16 for m in maps]):
        moved = au_pro_at_limits(transformed, masks, [0.05, 0.3])
        assert all(abs(base[k] - moved[k]) < 1e-12 for k in base)


def test_au_pro_weights_every_region_equally():
    """Two defects per image, one 25x the area of the other. A map that finds only the large one
    is excellent by pixel AUROC and mediocre by AU-PRO, which weights the two regions equally."""
    from sklearn.metrics import roc_auc_score

    masks = [np.zeros((64, 64), dtype=np.uint8) for _ in range(12)]
    for i in range(6):
        masks[i][8:28, 8:28] = 1               # 400 px
        masks[i][48:52, 48:52] = 1             # 16 px
    maps = [m.astype(np.float32) for m in masks]
    for m in maps:
        m[48:52, 48:52] = 0

    truth = np.concatenate([m.ravel() for m in masks])
    assert roc_auc_score(truth, np.concatenate([m.ravel() for m in maps])) > 0.97
    assert au_pro_at_limits(maps, masks, [0.3])[0.3] < 0.75


# --- VisA's split file ------------------------------------------------------------------------

# `Images/Normal/` holds train and test normals together, so only 1cls.csv tells them apart.
# These tests build a miniature VisA tree to check that the adapter reads the file.
VISA_ROWS = [
    ("candle", "train", "normal", "candle/Data/Images/Normal/0000.JPG", ""),
    ("candle", "train", "normal", "candle/Data/Images/Normal/0002.JPG", ""),
    ("candle", "test", "normal", "candle/Data/Images/Normal/0001.JPG", ""),
    # The same stem as the normal above, in the other directory, as in 5 of the 12 categories.
    ("candle", "test", "anomaly", "candle/Data/Images/Anomaly/0001.JPG",
     "candle/Data/Masks/Anomaly/0001.png"),
]


def _visa_root(tmp_path, rows=VISA_ROWS, pad=True):
    """A miniature VisA tree: a ``1cls.csv`` and an 8x8 file behind every row. ``pad`` adds rows
    for the categories ``rows`` leaves out, because the adapter requires all 12."""
    from PIL import Image

    from metacog.pipeline.datasets.visa import CATEGORIES

    rows = list(rows)
    if pad:
        for category in CATEGORIES:
            if not any(row[0] == category for row in rows):
                rows += [
                    (category, "train", "normal", f"{category}/Data/Images/Normal/000.JPG", ""),
                    (category, "test", "anomaly", f"{category}/Data/Images/Anomaly/000.JPG",
                     f"{category}/Data/Masks/Anomaly/000.png"),
                ]

    root = str(tmp_path)
    os.makedirs(os.path.join(root, "split_csv"), exist_ok=True)
    with open(os.path.join(root, "split_csv", "1cls.csv"), "w", newline="") as handle:
        handle.write("object,split,label,image,mask\n")
        for row in rows:
            handle.write(",".join(row) + "\n")
    for _, _, _, image, mask in rows:
        for relative in filter(None, (image, mask)):
            path = os.path.join(root, relative)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.new("L" if relative.endswith(".png") else "RGB", (8, 8)).save(path)
    return root


def _visa(root):
    from metacog.pipeline.datasets import VisA

    return VisA(root)


def test_visa_train_set_is_the_csv_not_the_normal_directory(tmp_path):
    """0001.JPG is in Normal/ but is a test image; listing the directory would fit on it."""
    train = _visa(_visa_root(tmp_path)).train_paths("candle")
    assert [os.path.basename(p) for p in train] == ["0000.JPG", "0002.JPG"]


def test_visa_test_items_are_ordered_and_masked(tmp_path):
    items = _visa(_visa_root(tmp_path)).test_items("candle")
    assert [(i.label, os.path.relpath(i.image_path, str(tmp_path))) for i in items] == [
        (0, "candle/Data/Images/Normal/0001.JPG"), (1, "candle/Data/Images/Anomaly/0001.JPG"),
    ]
    assert items[0].mask_paths == ()
    assert items[1].mask_paths[0].endswith("Masks/Anomaly/0001.png")


@pytest.mark.parametrize("rows,message", [
    ([("candle", "train", "anomaly", "candle/Data/Images/Anomaly/000.JPG",
       "candle/Data/Masks/Anomaly/000.png")] + VISA_ROWS, "one-class"),
    (VISA_ROWS[:3] + [("candle", "test", "anomaly", "candle/Data/Images/Anomaly/0001.JPG", "")],
     "mask is missing"),
    (VISA_ROWS + [("candle", "test", "normal", "candle/Data/Images/Normal/0000.JPG", "")],
     "both the train and test splits"),
])
def test_visa_split_file_is_validated(tmp_path, rows, message):
    with pytest.raises(AssertionError, match=message):
        _visa(_visa_root(tmp_path, rows)).train_paths("candle")


def test_visa_requires_the_split_file_to_cover_every_category(tmp_path):
    with pytest.raises(AssertionError, match="no train rows for 'capsules'"):
        _visa(_visa_root(tmp_path, pad=False)).train_paths("candle")


def test_visa_without_the_split_file_says_why_it_is_needed(tmp_path):
    os.makedirs(os.path.join(str(tmp_path), "candle"), exist_ok=True)
    with pytest.raises(AssertionError, match="1cls.csv must be present"):
        _visa(str(tmp_path)).train_paths("candle")


# --- configs --------------------------------------------------------------------------------

def _configs():
    for name in sorted(os.listdir(os.path.join(REPO, "configs"))):
        with open(os.path.join(REPO, "configs", name)) as handle:
            yield name, json.load(handle)


def test_every_config_names_a_registered_dataset_and_backbone():
    from metacog.pipeline.backbone import TIMM_MODELS
    from metacog.pipeline.datasets import DATASETS

    for name, config in _configs():
        assert config["dataset"]["name"] in DATASETS, name
        assert config["backbone"] in TIMM_MODELS, name
        assert geometry.resolve_scale(config["scale_rule"], (1024, 1024)) == 0.625, name


def test_every_config_is_scheduled_by_the_driver():
    """A config nobody runs drifts from the code without any test noticing."""
    import re

    configs = {os.path.splitext(name)[0] for name, _ in _configs()}
    with open(os.path.join(REPO, "run_metacog_grid.sh")) as handle:
        scheduled = set(re.findall(r"configs/([a-z0-9_]+)\.json", handle.read()))
    assert scheduled == configs, (
        f"scheduled but absent: {scheduled - configs}; present but unscheduled: {configs - scheduled}"
    )


# --- threshold rules ------------------------------------------------------------------------

def test_every_rule_is_derived_from_the_holdout_alone():
    from metacog.pipeline.thresholds import INHERITED, RULES, ORACLE, rules_from_holdout

    rng = np.random.default_rng(3)
    holdout = [rng.standard_normal((64, 64)).astype(np.float32) + 5.0 for _ in range(8)]
    thresholds = rules_from_holdout(holdout)

    assert set(thresholds) == set(RULES) - {ORACLE}
    pooled = np.concatenate([m.ravel() for m in holdout])
    assert thresholds["q95"] == pytest.approx(np.percentile(pooled, 95))
    assert thresholds["q99"] == pytest.approx(np.percentile(pooled, 99))
    assert thresholds["mu3sigma"] == pytest.approx(pooled.mean() + 3 * pooled.std(ddof=1))
    assert thresholds["inherited"] == pytest.approx(INHERITED * thresholds["q95"])
    # The EVT threshold extrapolates past what the holdout can count.
    assert thresholds["evt"] > thresholds["q99"]


def test_rules_refuse_an_empty_holdout():
    from metacog.pipeline.thresholds import rules_from_holdout

    with pytest.raises(AssertionError, match="no held-out normal maps"):
        rules_from_holdout([])


def test_f1max_is_the_best_gain_on_the_grid():
    """The oracle is a maximum over :data:`GAINS` on ``q95``, which contains 1.421, so it can never
    score below upstream's threshold. Rules anchored elsewhere can fall between grid points and
    beat it, so no bound over them is asserted."""
    from metacog.pipeline.evaluation import f1_at_threshold
    from metacog.pipeline.thresholds import GAINS, INHERITED, ORACLE, f1_for_rules, rules_from_holdout

    rng = np.random.default_rng(5)
    masks = [np.zeros((48, 48), dtype=bool) for _ in range(4)]
    masks[0][4:12, 4:12] = True
    masks[1][20:26, 20:26] = True
    maps = [rng.random((48, 48)).astype(np.float32) for _ in masks]
    holdout = [np.random.default_rng(6).random((48, 48)).astype(np.float32) for _ in range(4)]
    thresholds = rules_from_holdout(holdout)
    scored = f1_for_rules(maps, masks, thresholds, None, jobs=1)

    assert set(scored) == set(thresholds) | {ORACLE}
    assert all(0.0 <= v <= 1.0 for v in scored.values())
    assert INHERITED in GAINS and len(GAINS) == 64
    best = max(f1_at_threshold(maps, masks, thresholds["q95"] * gain) for gain in GAINS)
    assert scored[ORACLE] == best >= scored["inherited"]


def test_the_parallel_sweep_matches_the_serial_one():
    from metacog.pipeline.thresholds import f1_for_rules, rules_from_holdout

    rng = np.random.default_rng(8)
    masks = [rng.random((32, 32)) > 0.9 for _ in range(3)]
    maps = [rng.random((32, 32)).astype(np.float32) + m for m in masks]
    thresholds = rules_from_holdout([rng.random((32, 32)).astype(np.float32) for _ in range(3)])
    assert f1_for_rules(maps, masks, thresholds, None, jobs=1) == f1_for_rules(
        maps, masks, thresholds, None, jobs=4)


def test_the_morphology_radii_are_scaled_to_native():
    """``closing_radius`` and ``binary_erosion`` are lengths tuned at ``native // 4``, so they grow
    by 4 at native. ``closing_angles`` is a count and ``closing_lower_threshold`` a fraction."""
    from metacog.pipeline.evaluation import morphology_for_native

    native = morphology_for_native({"closing_radius": 26, "closing_angles": 16,
                                    "closing_lower_threshold": 0.8, "binary_erosion": 1,
                                    "evaluation_downscale": 4})
    assert (native.closing_radius, native.binary_erosion) == (104, 4)
    assert (native.closing_angles, native.closing_lower_threshold) == (16, 0.8)


def test_the_cross_validation_measures_the_fpr_the_evt_threshold_delivers():
    """On data with an exponential tail the GPD fit is exact, so the delivered FPR matches the
    target: 1.35e-3 at k = 3."""
    from metacog.pipeline.thresholds import alpha_k, cross_validated_fpr

    rng = np.random.default_rng(9)
    holdout = [rng.exponential(size=(200, 200)).astype(np.float32) for _ in range(10)]
    result = cross_validated_fpr(holdout, seed=42, ks=(3, 4))
    assert set(result) == {"3", "4"}
    assert all(r["pixels"] == 10 * 200 * 200 for r in result.values())
    assert result["3"]["alpha"] == pytest.approx(alpha_k(3)) == pytest.approx(1.35e-3, rel=0.01)
    assert result["3"]["exceed"] / result["3"]["pixels"] == pytest.approx(alpha_k(3), rel=0.25)
    assert result["4"]["exceed"] < result["3"]["exceed"]
    assert cross_validated_fpr(holdout, seed=42, ks=(3,)) == {"3": result["3"]}


def test_the_cross_validation_needs_one_image_per_fold():
    from metacog.pipeline.thresholds import cross_validated_fpr

    with pytest.raises(AssertionError, match="cannot be split into 5 folds"):
        cross_validated_fpr([np.zeros((4, 4), dtype=np.float32)] * 4, seed=42)


def test_the_paper_targets_553_pixels_at_k3_and_a_tenth_at_k5():
    from metacog.pipeline.thresholds import alpha_k

    assert alpha_k(3) * 640 * 640 == pytest.approx(553, abs=1)
    assert alpha_k(5) * 640 * 640 == pytest.approx(0.1, abs=0.02)
