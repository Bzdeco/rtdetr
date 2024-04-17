import math
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

from powerlines.data.utils import pad_array_to_match_target_size, MAX_DISTANCE_MASK_VALUE

CCQConfusionMatrix = namedtuple("CCQConfusionMatrix", ["tp", "fp", "fn"])

INVALID_MASK_VALUE = np.iinfo(np.uint16).max


def downsample(array: np.ndarray, downsampling_factor: int, min_pooling: bool = False) -> np.ndarray:
    input_dtype = array.dtype
    downsampler = nn.MaxPool2d(
        kernel_size=(downsampling_factor, downsampling_factor),
        stride=(downsampling_factor, downsampling_factor)
    )

    inversion = -1 if min_pooling else 1
    array = inversion * array

    padded_distance_mask = pad_array_to_match_target_size(
        array[np.newaxis, :, :], downsampling_factor, padding_value=array.min()
    )
    pooled = downsampler(torch.tensor(padded_distance_mask.astype(float)).float())[0]

    return (inversion * pooled.detach().cpu().numpy()).astype(input_dtype)


def distance_transform(
    binary_mask: np.ndarray, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if np.count_nonzero(binary_mask) == 0:
        side_size = binary_mask.shape[-1]
        max_distance = int(side_size * math.sqrt(side_size))
        distance_mask = np.full_like(binary_mask, fill_value=max_distance, dtype=float)

        if return_indices:
            indices = np.zeros((2,) + binary_mask.shape, dtype=int)
            return distance_mask, indices
        else:
            return distance_mask

    outside_binary_mask = np.logical_not(binary_mask)
    return distance_transform_edt(outside_binary_mask, sampling=1, return_distances=True, return_indices=return_indices)


def relaxed_confusion_matrix(
    pred_distance_mask: np.ndarray,
    target_distance_mask: np.ndarray,
    bin_thresholds: np.ndarray,
    tolerance_region: float
) -> CCQConfusionMatrix:
    num_thresh = len(bin_thresholds)
    # [num_thresholds, h, w]
    pred_bin_cable = np.repeat(pred_distance_mask[np.newaxis], num_thresh, axis=0) < bin_thresholds[:, np.newaxis, np.newaxis]
    gt_bin_cable = (target_distance_mask == 0)
    gt_distance = distance_transform(gt_bin_cable)

    # Compute distance mask from gt and predictions binarized with different thresholds
    pred_distance = np.zeros((num_thresh,) + pred_distance_mask.shape[-2:])
    for i in range(num_thresh):
        pred_distance[i] = distance_transform(pred_bin_cable[i])

    # Compute confusion matrix entries
    valid_image_area = np.repeat((target_distance_mask != INVALID_MASK_VALUE)[np.newaxis, :, :], num_thresh, axis=0)
    true_pos_area = np.logical_and(
        np.repeat(gt_distance[np.newaxis, :, :] <= tolerance_region, num_thresh, axis=0), valid_image_area
    )
    false_pos_area = np.logical_and(np.logical_not(true_pos_area), valid_image_area)
    false_neg_area = np.logical_and(pred_distance > tolerance_region, valid_image_area)

    tp = np.logical_and(true_pos_area, pred_bin_cable).sum(axis=(1, 2))
    fp = np.logical_and(false_pos_area, pred_bin_cable).sum(axis=(1, 2))
    fn = np.logical_and(false_neg_area, gt_bin_cable).sum(axis=(1, 2))

    return CCQConfusionMatrix(tp=tp, fp=fp, fn=fn)


def boxes_ccq_confusion_matrix(
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    bin_thresholds: np.ndarray,
    tolerance_region: float,
    downsampling_factor: int,
    mask_exclusion_zones: bool = False
) -> CCQConfusionMatrix:
    pred_distance_mask = boxes_to_downsampled_distance_mask(prediction["boxes"].numpy(), downsampling_factor)
    target_distance_mask = boxes_to_downsampled_distance_mask(target["boxes"].numpy(), downsampling_factor)

    if mask_exclusion_zones:
        exclusion_zone = downsample(target["exclusion_zone"].numpy(), downsampling_factor, min_pooling=False)
        target_distance_mask[exclusion_zone] = INVALID_MASK_VALUE

    return relaxed_confusion_matrix(
        pred_distance_mask,
        target_distance_mask,
        bin_thresholds,
        tolerance_region
    )


def visualize_distance_masks(pred_distance_mask: np.ndarray, target_distance_mask: np.ndarray, masked: bool):
    cmap = plt.get_cmap("gray").with_extremes(under="red", over="blue")
    combined = np.concatenate((pred_distance_mask, target_distance_mask), axis=1)

    folder = Path("/scratch/cvlab/home/gwizdala/output/distance_masks_temp")
    folder.mkdir(exist_ok=True)
    suffix = "_masked" if masked else ""
    filename = f"distance_mask_{len(list(folder.glob('*.png')))}{suffix}.png"

    print(np.count_nonzero(combined == INVALID_MASK_VALUE))
    combined[combined < 158] = 1 - np.clip(combined[combined < 158], a_min=0, a_max=128) / 128
    combined[combined >= 158] = 2
    plt.imshow(combined, cmap=cmap, vmin=0, vmax=1)
    plt.savefig(folder / filename)
    plt.close()


def boxes_to_downsampled_distance_mask(boxes: np.ndarray, downsampling_factor: int) -> np.ndarray:
    target_size = (94, 128)
    boxes_down = np.floor(boxes / downsampling_factor).astype(int)

    poles_mask = np.zeros(target_size, dtype=bool)
    for box in boxes_down:
        x0, y0, x1, y1 = box
        poles_mask[y0:(y1 + 1), x0:(x1 + 1)] = True

    return distance_transform(poles_mask)


EPS = 1e-8


def correctness(tp: Union[int, np.ndarray], fp: Union[int, np.ndarray]) -> Union[float, np.ndarray]:  # precision
    return tp / (tp + fp + EPS)


def completeness(tp: Union[int, np.ndarray], fn: Union[int, np.ndarray]) -> Union[float, np.ndarray]:  # recall
    return tp / (tp + fn + EPS)


def quality(
    tp: Union[int, np.ndarray], fp: Union[int, np.ndarray], fn: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    return tp / (tp + fp + fn + EPS)


@dataclass
class CCQMetric:
    def __init__(
        self, bin_thresholds: np.ndarray, tolerance_region: float, downsampling_factor: int, mask_exclusion_zones: bool = False
    ):
        self._bin_thresholds = bin_thresholds
        self._tolerance_region = tolerance_region
        self._downsampling_factor = downsampling_factor
        self._mask_exclusion_zones = mask_exclusion_zones

        n_thresholds = len(bin_thresholds)
        self._tp = np.zeros((n_thresholds,))
        self._fp = np.zeros((n_thresholds,))
        self._fn = np.zeros((n_thresholds,))

    def __call__(self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        conf_matrix = boxes_ccq_confusion_matrix(
            prediction,
            target,
            self._bin_thresholds,
            self._tolerance_region,
            self._downsampling_factor,
            self._mask_exclusion_zones
        )
        self._tp += conf_matrix.tp
        self._fp += conf_matrix.fp
        self._fn += conf_matrix.fn

    def compute(self) -> Dict[str, float]:
        results = {}

        corr = correctness(self._tp, self._fp)
        compl = completeness(self._tp, self._fn)
        qual = quality(self._tp, self._fp, self._fn)
        thresholds = self._bin_thresholds / MAX_DISTANCE_MASK_VALUE

        for metric_name, values in zip(["correctness", "completeness", "quality"], [corr, compl, qual]):
            for i, threshold in enumerate(thresholds):
                results[f"{metric_name}/{threshold:.3f}"] = values[i]

        return results
