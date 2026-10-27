"""The downscale applied before tiling, and the tile layout it produces.

SuperADD's paper says *"images are downscaled by a factor of 0.625"*. Its code computes the factor
as ``patch_size / 1024`` (``train.py:28``), so that one 640 px tile covers 1024 native pixels.
Upstream never pads, and ``PatchedExecution.axis_patch_split`` asserts ``dim_size >= patch_size``
after the downscale (``model.py:55``). A fixed 0.625 is therefore illegal on any image whose short
side is under 1024 px: 12 of 15 MVTec AD categories and 5 of 12 VisA categories.

The rule used everywhere is ``max(scale, min_side / short_side)`` with ``scale = 0.625`` and
``min_side = 640``: upstream's factor wherever it is legal, and otherwise the smallest factor that
is. On MVTec AD 2 it is exactly 0.625. On MVTec AD, whose images are all square, it lands every
category on one 640 x 640 tile. On VisA it keeps every category at 24.0-26.1 native pixels per
token, against MVTec AD 2's 25.6.

The layout numbers are computed by calling upstream's own ``PatchedExecution.axis_patch_split``,
so the geometry printed and recorded before a run is the one the encoder gets.
"""
from __future__ import annotations

from industrial.model import PatchedExecution

from .backbone import PATCH_SIZE

#: SuperADD's ``patch_size``: the side of one encoder tile, in downscaled pixels.
TILE_SIZE = 640

#: SuperADD's ``patch_overlap``. It is a lower bound on the context margin, not the overlap you
#: get: the tile count is derived from it and the tile starts are then spread evenly.
TILE_OVERLAP = 128


def resolve_scale(rule: dict, native_hw: tuple[int, int]) -> float:
    """``max(scale, min_side / short_side)``. See the module docstring."""
    scale, min_side = float(rule["scale"]), int(rule["min_side"])
    assert 0.0 < scale <= 1.0 and min_side > 0, f"bad scale rule {rule}"
    return max(scale, min_side / min(native_hw))


def downscaled_size(native_hw: tuple[int, int], scale: float) -> tuple[int, int]:
    """SuperADD's downscaled size, truncated with ``int()`` exactly as ``model.py:37`` does.

    Do not round this to the patch grid. The encoder reads ``dim // 16`` tokens, so the pixels past
    the last whole patch are dropped (up to 11 px on MVTec AD 2's ``vial``), and rounding instead
    changes the token grid on 5 of the 8 MVTec AD 2 categories.
    """
    return int(native_hw[0] * scale), int(native_hw[1] * scale)


def assert_geometry_is_legal(native_hw: tuple[int, int], scale: float, tile_size: int = TILE_SIZE) -> None:
    """Raise upstream's ``dim_size >= patch_size`` condition before the encoder runs, with a reason.

    Upstream asserts it without a message, from inside ``PatchedExecution``, on the first image.
    """
    height, width = downscaled_size(native_hw, scale)
    assert min(height, width) >= tile_size, (
        f"native {native_hw[0]}x{native_hw[1]} at scale {scale:.6f} downscales to {height}x{width}, "
        f"whose short side is below the {tile_size} px tile, and SuperADD does not pad "
        f"(model.py:55). Set scale_rule.min_side to {tile_size}."
    )


def axis_tiles(dim_size: int, tile_size: int, tile_overlap: int, patch: int) -> tuple[list, list]:
    """``(input_rois, prediction_rois)`` along one axis, from upstream's own arithmetic."""
    input_rois, prediction_rois, _ = PatchedExecution(tile_size, tile_overlap, patch).axis_patch_split(dim_size)
    return input_rois, prediction_rois


def context_margin(dim_size: int, tile_size: int, tile_overlap: int, patch: int) -> int:
    """The fewest tokens trimmed at any interior tile boundary along one axis.

    This is the quantity ``tile_overlap`` bounds from below: every kept token had at least this many
    tokens of context in its own forward pass. With one tile there is no seam, and it is the tile.
    """
    _, prediction_rois = axis_tiles(dim_size, tile_size, tile_overlap, patch)
    tile_tokens = min(tile_size, dim_size) // patch
    if len(prediction_rois) == 1:
        return tile_tokens
    trims = [start for start, _ in prediction_rois[1:]]
    trims += [tile_tokens - end for _, end in prediction_rois[:-1]]
    return min(trims)


def summary(
    native_hw: tuple[int, int],
    scale: float,
    tile_size: int = TILE_SIZE,
    tile_overlap: int = TILE_OVERLAP,
    patch: int = PATCH_SIZE,
) -> dict:
    """One category's layout, for the log line and the results JSON."""
    height, width = downscaled_size(native_hw, scale)
    grid_h, grid_w = height // patch, width // patch
    tiles_y = len(axis_tiles(height, tile_size, tile_overlap, patch)[0])
    tiles_x = len(axis_tiles(width, tile_size, tile_overlap, patch)[0])
    return {
        "native_hw": list(native_hw),
        "scale": scale,
        "downscaled_hw": [height, width],
        "dropped_px": [height % patch, width % patch],
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tiles_per_axis": [tiles_y, tiles_x],
        "n_tiles": tiles_y * tiles_x,
        "grid_hw": [grid_h, grid_w],
        "n_tokens": grid_h * grid_w,
        "context_margin": [
            context_margin(height, tile_size, tile_overlap, patch),
            context_margin(width, tile_size, tile_overlap, patch),
        ],
        # How many native pixels one token summarizes.
        "native_px_per_token": [native_hw[0] / grid_h, native_hw[1] / grid_w],
    }
