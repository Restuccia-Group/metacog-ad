"""The DINOv3 encoders, loaded through timm instead of upstream's ``torch.hub`` loader.

Upstream's ``DinoV3Backbone`` (``model.py:126-140``) loads Meta's gated checkpoint release through
``torch.hub``. timm publishes the same LVD-1689M weights on the Hugging Face Hub, so they are loaded
from there.

Upstream calls ``get_intermediate_layers(x, n=layers, norm=False)``: the raw output of the tapped
blocks, with no final norm and no prefix tokens. timm's DINOv3 class does not have that method; its
equivalent is ``forward_intermediates(x, indices=layers, norm=False, output_fmt='NLC',
intermediates_only=True)``, which returns the same ``[B, T, C]`` tensors in the same order.
``tests/test_backbone.py`` checks this at every size.

All four sizes are patch 16 and use RoPE rather than a learned position embedding, so they share
the tiling geometry and any tile size is valid. Everything runs in float32 with TF32 off, as in
upstream.
"""
from __future__ import annotations

import torch

#: The DINOv3 family, keyed by upstream's ``backbone`` config string.
TIMM_MODELS = {
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_vith16plus": "vit_huge_plus_patch16_dinov3.lvd1689m",
}

#: The patch stride of every model above. ``run.py`` needs it to report the tile geometry
#: before the encoder is built; :class:`TimmDinoV3Backbone` checks it against the built model.
PATCH_SIZE = 16


class TimmDinoV3Backbone:
    """Drop-in for upstream's ``DinoV3Backbone``: a callable returning one tensor per tapped layer."""

    def __init__(self, model_name: str, layers: list[int], device, pretrained: bool = True) -> None:
        import timm

        assert model_name in TIMM_MODELS, (
            f"unknown backbone {model_name!r}; available: {sorted(TIMM_MODELS)}"
        )
        self.layers = list(layers)
        # `dynamic_img_size` costs nothing here: DINOv3's RoPE is normalized to the grid and there is
        # no absolute position embedding to resample.
        model = timm.create_model(
            TIMM_MODELS[model_name], pretrained=pretrained, num_classes=0, dynamic_img_size=True
        )
        self.dino = model.eval().to(device)
        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)

        self.model_patch_size = self.dino.patch_embed.patch_size[0]
        assert self.model_patch_size == PATCH_SIZE, (
            f"{model_name} has patch {self.model_patch_size}, not {PATCH_SIZE}"
        )
        self.embed_dim = int(self.dino.embed_dim)
        self.n_blocks = len(self.dino.blocks)
        assert max(self.layers) < self.n_blocks, (
            f"{model_name} has {self.n_blocks} blocks; layer indices {self.layers} are 0-based"
        )

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.inference_mode():
            return self.dino.forward_intermediates(
                x, indices=self.layers, norm=False, output_fmt="NLC", intermediates_only=True
            )
