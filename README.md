# A Metacognitive Perspective on Anomaly Detection with Vision Foundation Models

This repository holds the code to reproduce the results of the paper. It uses a frozen DINOv3 encoder and
changes the anomaly scorer on top of it.
The two anomaly scorers that are used are a nearest-neighbour memory bank, and a Mahalanobis scorer that fits one Gaussian for each layer.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{galiazzo2026metacognitive,
  title={A Metacognitive Perspective on Anomaly Detection with Vision Foundation Models},
  author={Galiazzo, Matteo and Onorato, Gabriele and Loi, Dario and Bastian, Nathaniel D. and Restuccia, Francesco},
  booktitle={AAAI Conference on Artificial Intelligence (METACOG Workshop)},
  year={2026}
}
```

## Core idea

A metacognitive view of Anomaly Detection exposes two problems:

- SOTA Anomaly Detection pipelines couple a frozen VFM (object level) and a scorer (meta level), so
their individual contributions are unclear.
- Even if the monitoring signal improves, the control (commitment to a decision) can degrade, showing
that the two functions can dissociate.

The threshold is the cause of the dissociation.
The $\mu + 3 \sigma$ rule suggested by the MVTec AD 2 dataset assumes that the score tail is Gaussian, but the real
tails of the distribution are heavier, thus the rule admits more false positives than it promises.

We fit the tail by using Extreme Value Theory, and then we set the threshold to a fixed number of
false-positive pixels. This process doesn't use anomalous data.
On MVTec AD 2 the segmentation $F_1$ difference between the two scorers falls
from 13.46 to 0.61 points. On MVTec AD and VisA the Mahalanobis scorer is better than the nearest-neighbor one by 10.78 and 8.74 points.

## Repository structure

- `src/metacog/scorers/`: the two scorers. `base.py` is the interface, `knn.py` the memory
  bank, `mahalanobis.py` the Mahalanobis scorer.
- `src/metacog/pipeline/`: everything both scorers share: the DINOv3 encoder, the scale rule and
  tiling, the three datasets, the threshold rules (`thresholds.py`) and the metrics.
- `src/metacog/run.py`: the only command. It fits and scores both scorers on one dataset,
  derives every threshold rule from the held-out normal images, cross-validates the EVT threshold's
  false-positive rate on them, and writes one results file.
- `tables/tables_seeds.py`: writes Tables 1 and 2 as LaTeX. `tables/figure2.py` writes the data of
  Figure 2, with the FLOPs counted on CPU.
- `configs/`: `metacog_{ad,ad2,visa}.json` produce the two tables; `metacog_ad2_vit{s,b,l}.json`
  change only the encoder, for Figure 2.
- `run_metacog_grid.sh`: runs the grid on one or more GPUs; `merge_shards.py` joins the parts.
- `tracks/industrial/`: the SuperADD code. `docs/UPSTREAM_DIFF.md` lists every change to it.

## Datasets

Get MVTec AD, MVTec AD 2 and VisA from their own websites.
The directory of the dataset is given to each script with `--data-root`.

## Install

This project uses [uv](https://docs.astral.sh/uv/). `uv sync` reads `pyproject.toml` and
`uv.lock`, makes a virtual environment in `.venv/`, and installs the exact versions in the lock
file.

```bash
uv sync
uv run pytest tests/
```

`uv run <command>` then runs that command in the environment.

timm downloads the DINOv3 weights on the first run and keeps them in
`~/.cache/huggingface`.

If you prefer pip, `pip install -e .` works too.

## Quick start

1. Run the grid. This needs a GPU. It writes the results to `$RUN_ROOT`. Give it the directory of
   each dataset that you run, and the GPUs to use (the default is GPU 0).

```bash
RUN_ROOT=/path/with/space \
MVTEC_AD_ROOT=/path/to/mvtec_ad \
MVTEC_AD_2_ROOT=/path/to/mvtec_ad_2 \
VISA_ROOT=/path/to/visa \
GPUS="0 1" ./run_metacog_grid.sh
```

Each run derives its own thresholds and scores the F1 of every rule, so there is no second stage.
The oracle sweep is one morphology pass for each of 64 gains over every test image, which is the
longest CPU step; it uses every core by default, and `--f1-workers` sets the count. The encoder
sweep for Figure 2 is not in the default grid: add `DATASETS="ad2_vits ad2_vitb ad2_vitl"`.

2. Write the tables.

```bash
METACOG_RUN_DIR=$RUN_ROOT tables/run_all.sh
```

It runs `tables_seeds.py` and `figure2.py`, which read no image, only
`$METACOG_RUN_DIR/results/<ds>/seed<N>/<ds>.json` with
`<ds>` in `ad`, `ad2`, `visa` (and `ad2_vits`, `ad2_vitb`, `ad2_vitl` for Figure 2) and `<N>` in 42,
43, 44 (`METACOG_SEEDS` overrides the list), and write to `tables/out/`:

| output | paper |
|---|---|
| `seeds_summary_free.tex` | **Table 2**: AU-PRO and image AUROC per dataset |
| `seeds_summary_rules.tex` | **Table 1**: segmentation F1 at Q99 / µ+3σ / EVT, and the F1max oracle |
| `seeds_aupro_{ad,ad2,visa}.tex` | the per-category numbers behind Table 2 |
| `seeds_rules_{ad,ad2,visa}.tex` | the per-category numbers behind Table 1 |
| `fpr_cv.tex` | the cross-validation behind k = 4: false-positive pixels per 640 × 640 normal image the EVT threshold delivers on held-out normals, for k = 3, 4, 5 |
| `figure2.csv` | **Figure 2**: FLOPs per MVTec AD 2 test image, AU-PRO@0.05 and F1 at the EVT threshold, per encoder and scorer |

A missing cell prints as `---`. A cell with a ± is the mean over seeds ± the sample standard
deviation. The seed drives the coreset subsample and the brightness-augmentation stream; the
backbone is frozen and the Mahalanobis fit takes no seed, so the spread is noise in the fit. With
three seeds it is itself noisy, and it is not a confidence interval.

`figure2.py` counts the encoder FLOPs with `FlopCounterMode` on one CPU forward pass per encoder. It
builds each encoder with random weights, since the count does not depend on them, so it downloads
nothing.

## Differences from the SuperADD evaluation

- SuperADD's `test.py` skips the anomaly-free test images and resizes the ground truth to the
  model's output. We score every test image at native resolution, against the ground truth as
  published.
- SuperADD downscales every image by 0.625, so that one 640 px tile covers 1024 native pixels. On
  MVTec AD and VisA many images have a short side under 1024 px, where 0.625 would make the image
  smaller than one tile. There we use the smallest factor that keeps the short side at 640 px. On
  MVTec AD 2 the factor is 0.625, as in SuperADD.

## Licence

**CC-BY-NC-4.0**, from SuperADD. This includes the new code in `src/metacog/`. See [`NOTICE`](NOTICE)
for each file and its licence. The MVTec AU-PRO code is BSD-3-Clause. The DINOv3 weights have their
own terms from Meta.
