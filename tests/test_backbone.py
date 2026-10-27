"""The timm loader against the call upstream makes, on real weights, on CPU.

``backbone.py`` replaces ``get_intermediate_layers(x, n=layers, norm=False)`` with timm's
``forward_intermediates``. A normed or off-by-one tap would still give a plausible anomaly map, so
the equivalence is tested rather than assumed.

timm downloads the weights (330 MB and up) on the first run. The tests are skipped when the download
is not possible.
"""
from __future__ import annotations

import pytest
import torch

from metacog.pipeline.backbone import PATCH_SIZE, TIMM_MODELS, TimmDinoV3Backbone

# (backbone, taps at 25/50/75/100% depth, width, depth), at the 640 px tile every config runs.
DINOV3_ROSTER = [
    ("dinov3_vits16", [2, 5, 8, 11], 384, 12),
    ("dinov3_vitb16", [2, 5, 8, 11], 768, 12),
    ("dinov3_vitl16", [5, 11, 17, 23], 1024, 24),
    ("dinov3_vith16plus", [7, 15, 23, 31], 1280, 32),
]
ROSTER_IDS = [name for name, *_ in DINOV3_ROSTER]
TILE = 640


def _build(name, layers):
    try:
        return TimmDinoV3Backbone(name, layers, torch.device("cpu"))
    except Exception as error:  # noqa: BLE001 -- missing weights is the only case worth skipping
        pytest.skip(f"{TIMM_MODELS[name]} unavailable ({type(error).__name__}): {error}")


@pytest.mark.parametrize("name,layers,width,depth", DINOV3_ROSTER, ids=ROSTER_IDS)
def test_the_model_matches_its_roster_entry(name, layers, width, depth):
    backbone = _build(name, layers)
    assert backbone.model_patch_size == PATCH_SIZE
    assert backbone.embed_dim == width
    assert backbone.n_blocks == depth


@pytest.mark.parametrize("name,layers,width,depth", DINOV3_ROSTER, ids=ROSTER_IDS)
def test_four_taps_come_back_distinct_and_without_prefix_tokens(name, layers, width, depth):
    """DINOv3 has 5 prefix tokens (CLS and four registers). If they came back, they would enter
    the fit as patch tokens."""
    backbone = _build(name, layers)
    outs = backbone(torch.randn(1, 3, TILE, TILE))
    assert len(outs) == len(layers)
    for out in outs:
        assert out.shape == (1, (TILE // PATCH_SIZE) ** 2, width)
    for shallower, deeper in zip(outs, outs[1:]):
        assert not torch.allclose(shallower, deeper), "two taps returned the same tensor"


@pytest.mark.parametrize("name,layers,width,depth", DINOV3_ROSTER, ids=ROSTER_IDS)
def test_the_deepest_tap_is_the_last_block_before_the_norm(name, layers, width, depth):
    """``norm=False`` matters: the final norm would rescale every token before the fit sees it."""
    backbone = _build(name, layers)
    x = torch.randn(1, 3, TILE, TILE)
    deepest = backbone(x)[-1]
    with torch.inference_mode():
        normed = backbone.dino.forward_features(x)[:, backbone.dino.num_prefix_tokens:]
        assert torch.allclose(backbone.dino.norm(deepest), normed, atol=1e-5), (
            "the deepest tap is not the last block: forward_features disagrees after its norm"
        )
    assert not torch.allclose(deepest, normed, atol=1e-3), "the tap came back already normed"
