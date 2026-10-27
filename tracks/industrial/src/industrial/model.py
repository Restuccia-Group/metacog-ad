import torch
import gc
import numpy as np
from math import ceil, floor
from itertools import product
from tqdm import tqdm
from collections import defaultdict
import sys
# metacog: the kNN primitives move behind the Scorer interface (metacog/scorers/knn.py), and a
# timm loader replaces DinoV3Backbone, which removes the industrial.paths import.
# nearest_neighbor.py itself is unchanged.
from metacog.pipeline.backbone import TimmDinoV3Backbone
from metacog.scorers import build_scorer
from industrial.post_process import multi_oriented_closing, erosion_on_binary_maps, fill_closed_regions


class PreProcessing:

    def __init__(self, device, resize_factor, brightness_augmentation = (1, 1), normalization='imagenet'):
        self.device = device
        self.resize_factor = resize_factor
        self.min_brightness_factor, self.max_brightness_factor = brightness_augmentation

        if normalization is None:
            self.norm_mean = torch.tensor([0.0, 0.0, 0.0], device=device)[:, None, None]
            self.norm_std = torch.tensor([1.0, 1.0, 1.0], device=device)[:, None, None]
        elif normalization == 'imagenet':
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
        else:
            raise ValueError(f'Invalid normalization {normalization}.')


    def __call__(self, x: torch.Tensor):
        assert x.ndim == 4
        factor = np.random.uniform(self.min_brightness_factor, self.max_brightness_factor)
        x = torch.clip(x * factor, 0, 1)
        new_shape = (int(x.shape[-2] * self.resize_factor), int(x.shape[-1] * self.resize_factor))
        x = torch.nn.functional.interpolate(x, size=new_shape, mode='bicubic', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        return x

class PatchedExecution:

    def __init__(self, patch_size: int, patch_overlap: int, model_patch_size: int):
        assert patch_overlap > 0
        assert patch_size > 2 * patch_overlap
        assert patch_size % model_patch_size == 0
        assert patch_overlap % model_patch_size == 0

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.model_patch_size = model_patch_size

    def axis_patch_split(self, dim_size):
        assert dim_size >= self.patch_size

        dim_size_t = dim_size // self.model_patch_size
        overlap_t = self.patch_overlap // self.model_patch_size
        patch_size_t = self.patch_size // self.model_patch_size

        n_patches = ceil((dim_size_t - patch_size_t) / (patch_size_t - 2 * overlap_t)) + 1

        fac = dim_size_t - patch_size_t
        div = max(1, n_patches - 1)

        input_rois = []
        for i in range(n_patches):
            patch_min = (i * fac) // div
            patch_max = patch_min + patch_size_t
            input_rois.append((patch_min, patch_max))

        prediction_rois = []
        for i in range(n_patches):
            start = 0 if i == 0 else ceil((input_rois[i - 1][1] - input_rois[i][0]) / 2)
            end = patch_size_t if i == n_patches - 1 else patch_size_t - floor((input_rois[i][1] - input_rois[i + 1][0]) / 2)
            prediction_rois.append((start, end))

        result_rois = [(0, prediction_rois[0][1] - prediction_rois[0][0])]
        for i in range(1, n_patches):
            start = result_rois[i - 1][1]
            end = start + prediction_rois[i][1] - prediction_rois[i][0]
            result_rois.append((start, end))

        input_rois = [(s * self.model_patch_size, e * self.model_patch_size) for (s, e) in input_rois]

        return input_rois, prediction_rois, result_rois

    def __call__(self, x: torch.Tensor, func: callable):
        assert len(x.shape) == 4

        b, c, h, w = x.shape
        patch_w, patch_h = self.patch_size, self.patch_size

        input_rois_y, prediction_rois_y, result_rois_y = self.axis_patch_split(h)
        input_rois_x, prediction_rois_x, result_rois_x = self.axis_patch_split(w)

        x_overlapped = torch.zeros((b, len(input_rois_y) * len(input_rois_x), c, patch_h, patch_w), device=x.device)

        for i, ((y_start, y_end), (x_start, x_end)) in enumerate(product(input_rois_y, input_rois_x)):
            x_overlapped[:, i] = x[:, :, y_start:y_end, x_start:x_end]

        x_overlapped = x_overlapped.reshape(-1, c, patch_h, patch_w)

        prediction = func(x_overlapped)

        tokens_y, tokens_x = patch_h // self.model_patch_size, patch_w // self.model_patch_size

        prediction = torch.stack(prediction)
        vector_count, _, _, feature_count = prediction.shape

        prediction = prediction.reshape(vector_count, b, -1, tokens_y, tokens_x, feature_count)

        result = torch.zeros((vector_count, b, h // self.model_patch_size, w // self.model_patch_size, feature_count), device=prediction.device)

        prediction_rois = product(prediction_rois_y, prediction_rois_x)
        result_rois = product(result_rois_y, result_rois_x)
        for i, ((pred_roi_y, pred_roi_x), (res_roi_y, res_roi_x)) in enumerate(zip(prediction_rois, result_rois)):
            p = prediction[:, :, i, pred_roi_y[0]:pred_roi_y[1], pred_roi_x[0]:pred_roi_x[1]]
            result[:, :, res_roi_y[0]:res_roi_y[1], res_roi_x[0]:res_roi_x[1]] = p

        result = [t.cpu().numpy() for t in result]

        return result


# metacog: DinoV3Backbone deleted. It loaded Meta's gated .pth through torch.hub;
# metacog.pipeline.backbone.TimmDinoV3Backbone loads the same LVD-1689M weights through timm and
# taps the same blocks with norm=False. See docs/UPSTREAM_DIFF.md.

class PostProcessing:

    def __init__(self, closing_radius: int, closing_angles: int, closing_lower_threshold: float, binary_erosion: int):
        self.closing_radius = closing_radius
        self.closing_angles = closing_angles
        self.closing_lower_threshold = closing_lower_threshold
        self.binary_erosion = binary_erosion

    def __call__(self, x: np.ndarray, threshold):
        x = multi_oriented_closing(x, threshold, self.closing_radius, self.closing_angles, self.closing_lower_threshold)
        x = fill_closed_regions(x)
        x = erosion_on_binary_maps(x, self.binary_erosion)
        return x


class SuperADD:

    def __init__(self, backbone: str, layers: list[int], threshold_fraction: int, resize_factor: float,
                 patch_size: int, patch_overlap: int, max_database_size: int, subsampling_iterations: int,
                 threshold_percentile: float, threshold_factor: float, evaluation_downscale: int,
                 closing_radius: int, closing_angles: int, closing_lower_threshold: float, binary_erosion: int,
                 brightness_augmentation = (1, 1), device='cuda',
                 scorer: str = 'knn', scorer_config: dict | None = None):   # metacog
        self.backbone_name = backbone
        self.threshold_fraction = threshold_fraction
        self.device = device
        self.layers = layers
        self.augmented_preprocessing = PreProcessing(device, resize_factor, brightness_augmentation)
        self.preprocessing = PreProcessing(device, resize_factor, (1., 1.))
        self.backbone = TimmDinoV3Backbone(backbone, layers, device)   # metacog: timm, not torch.hub
        self.patch_exec = PatchedExecution(patch_size, patch_overlap, self.backbone.model_patch_size)
        self.max_database_size = max_database_size
        self.subsampling_iterations = subsampling_iterations
        self.threshold_percentile = threshold_percentile
        self.threshold_factor = threshold_factor

        # metacog: the memory bank is now one of two Scorer implementations.
        self.scorer = build_scorer(scorer, layers, device, **(scorer_config or {}))
        self.threshold = 0

        # metacog: if set, predict passes every anomaly map it produces to this callable.
        # metacog/run.py uses it to keep the held-out normal maps that train scores to fit the
        # threshold and then drops, because every threshold rule is derived from them.
        self.map_sink = None

        self.evaluation_downscale = evaluation_downscale

        self.postprocessing = PostProcessing(closing_radius, closing_angles, closing_lower_threshold, binary_erosion)

    def __clear_cache(self):
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        gc.collect()

    def train(self, train_data: list[torch.Tensor]):

        train_data_prototypes = [t for i, t in enumerate(train_data) if i % self.threshold_fraction != 0]
        train_data_threshold = [t for i, t in enumerate(train_data) if i % self.threshold_fraction == 0]

        self.__clear_cache()

        # metacog: the loop is unchanged; each [T, C] block now goes to scorer.partial_fit. The kNN
        # scorer keeps the blocks for its coreset, the Mahalanobis scorer folds them into a Gram.
        for x in tqdm(train_data_prototypes, 'processing prototype train data', file=sys.stdout):
            x = x.to(self.device)[None]
            x = self.augmented_preprocessing(x)
            prediction = self.patch_exec(x, self.backbone)
            for layer, embedding in zip(self.layers, prediction):
                self.scorer.partial_fit(layer, embedding.reshape(-1, embedding.shape[-1]))

        self.scorer.finalize()
        self.__clear_cache()

        anomaly_maps = []
        for x in tqdm(train_data_threshold, 'processing threshold train data', file=sys.stdout):
            anomaly_map, _ = self.predict(x)
            anomaly_maps.append(anomaly_map)

        self.threshold = np.percentile(anomaly_maps, self.threshold_percentile) * self.threshold_factor

        print(f'auto-detected threshold {self.threshold:.3f}')


    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        input_shape = x.shape
        assert len(input_shape) == 3 and input_shape[0] == 3

        output_shape = (input_shape[-2] // self.evaluation_downscale, input_shape[-1] // self.evaluation_downscale)

        x = x.to(self.device)[None]

        x = self.preprocessing(x)
        predicted_embeddings = self.patch_exec(x, self.backbone)

        layer_distances = []
        for layer, predicted_embedding in zip(self.layers, predicted_embeddings):
            _, h, w, c = predicted_embedding.shape
            query = torch.as_tensor(predicted_embedding).reshape(h * w, c).to(self.device)
            # metacog: was nearest_neighbors(query, keys, knn_neighbors=1).mean(dim=-1).
            dists = self.scorer.distance(layer, query)
            dists = dists.reshape(h, w) / c # normalize distance by dimsize to account for different dimsizes
            layer_distances.append(dists)

        self.__clear_cache()

        layer_distances = torch.stack(layer_distances, dim=0)
        layer_distances = torch.nn.functional.interpolate(layer_distances[None], size=output_shape, mode='bilinear', align_corners=False)[0]
        anomaly_map = torch.mean(layer_distances, dim=0)
        anomaly_map = anomaly_map.cpu().numpy()
        if self.map_sink is not None:   # metacog: see __init__
            self.map_sink(anomaly_map)

        binary_result = self.postprocessing(anomaly_map, self.threshold)

        self.__clear_cache()

        return anomaly_map, binary_result
