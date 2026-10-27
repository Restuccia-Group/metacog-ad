# Changes to the SuperADD code

This fork starts from SuperADD (`github.com/LukasRoom/SuperADD`, commit `44cf251`).
This file lists every change that we made to that code.

`tracks/` holds the SuperADD files. `src/metacog/` holds the new code. All new behaviour is in
`src/metacog/`. There is one exception: `model.py`. It connects the two, and this file lists each
change to it.

`industrial.model` imports `metacog.pipeline.backbone` and `metacog.scorers`. Neither of them
imports `industrial.model`, so there is no import loop.

## Files that did not change

These two files must stay the same as SuperADD. Their only value is to give the same numbers.
`tests/test_core.py` compares their sha256 values on each test run.

| file | sha256 |
|---|---|
| `tracks/industrial/src/industrial/nearest_neighbor.py` | `a0daeb8d…d87f5` |
| `tracks/industrial/src/industrial/post_process.py` | `3150b5eb…d5585` |

## The nine changes to model.py

Each change has a `# metacog:` comment in the code. The line numbers below are the line numbers of
the **SuperADD** file, not of the file in this repository.

| # | line | change | reason |
|---|---|---|---|
| 1 | `:6`, `:10`, `:12-13` | Remove the `industrial.paths`, `nearest_neighbor`, `pathlib` and `json` imports. Import `TimmDinoV3Backbone` and `build_scorer`. | A result of changes 2, 5 and 8. |
| 2 | `:126-140` | Delete the `DinoV3Backbone` class. | It loads Meta's gated weights file through `torch.hub`. timm has the same LVD-1689M weights. |
| 3 | `:159-163` | Add the `scorer` and `scorer_config` arguments. | They select the arm. |
| 4 | `:170` | Build a `TimmDinoV3Backbone`. | See change 2. |
| 5 | `:177` | Replace the memory bank with a `Scorer` object. | The memory bank becomes one of two scorers. |
| 6 | `:196-211` | Send each token block to `scorer.partial_fit()`. | The loop does not change. The Mahalanobis arm reads the tokens once and keeps only a small matrix. |
| 7 | `:238-241` | Replace the four lines of kNN code with `scorer.distance()`. | This is the main change. |
| 8 | `:257-299` | Delete `to_disk` and `from_disk`. | No run saves or loads a model. They stored the memory bank, which the Mahalanobis arm does not have. |
| 9 | after `:178`, after `:249` | Add `map_sink`, an optional callable that receives each anomaly map `predict` produces. | `train` scores the held-out normal images to fit its threshold, then drops the maps. Every threshold rule is derived from them. When `map_sink` is `None`, `predict` is upstream's. |

## What did not change

This list is the reason that the results are comparable to the SuperADD results.

- `:191-192` — the 7/8 prototype and 1/8 threshold split
- `:199` and `:231` — brightness augmentation during the fit, but not during inference
- `:218` — the threshold `np.percentile(maps, 95) * 1.421`
- `:227` — `evaluation_downscale = 4`, on the native image size
- `:241` — `dists / c`
- `:247-248` — bilinear resize to `native // 4`, then the mean over the layers
- all of `PatchedExecution`, and all of `PreProcessing`

## SuperADD behaviour that we keep on purpose

Each item below looks like a defect. We keep all of them, because the results must stay comparable
to the SuperADD results. Do not "correct" them.

- **`PreProcessing` removes up to 11 pixels on each axis.** `int(side * scale)` does not align to
  the patch grid, and the encoder reads `dim // 16` tokens, so it never sees the last pixels. We
  keep this arithmetic exactly: aligning to the patch grid changes the token grid on 5 of the 8
  MVTec AD 2 categories, and the two runs would then see different pixels. Each category reports
  the count as `geometry.dropped_px`.
- **`PreProcessing(normalization=None)` causes an `AttributeError`.** The `None` branch sets
  `norm_mean` and `norm_std`, but `__call__` reads `mean` and `std`. No code uses `None`.
- **`subsampling_iterations` is a dead key.** SuperADD stores it but passes a hardcoded 100 to the
  subsampling. Here `metacog/run.py` passes the config value to `KnnScorer`; every config sets it to
  100, so the behaviour is the same.

## Deleted SuperADD files

`NOTICE` gives the licence of each file. We deleted `utils/industrial/evaluate.py` (no code calls
it), `utils/industrial/submission.py` (challenge packaging), `utils/industrial/types.py` and
`dataset.py` (they support only MVTec AD 2), `utils/industrial/paths.py`, `utils/auto_batch.py`,
the `train.py`, `test.py` and `submit.py` scripts (`src/metacog/run.py` replaces them), `config.json`
(now one file for each run), `uv.lock`, and the two sub-project `pyproject.toml` files.
One `pyproject.toml` at the root replaces them.

## Two differences from the SuperADD run

1. **The weights come from timm, not from the Meta release.** They are the same LVD-1689M
   checkpoint in a different package. Each results file names the timm model as
   `config.backbone_timm`.
2. **The random draw order in `subsampling_distance_based_fast` can differ.** This occurs if the
   SuperADD process used the random number generator in a different order before this point. The
   seed (42) and the algorithm are theirs.
