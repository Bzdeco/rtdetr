from typing import Dict, Tuple, Union

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from powerlines.ccq import CCQMetric
from powerlines.data.utils import MAX_DISTANCE_MASK_VALUE


def mean_average_precision():
    # Must be used together with the postprocessor
    return MeanAveragePrecision(
        box_format="xyxy",
        average="macro",
        backend="faster_coco_eval"
    )


def ccq(downsampling_factor: int, mask_exclusion_zones: bool, tolerance_region: float = 1.42):
    return CCQMetric(
        # standard + default + NEVBW 2 cells distance, rescaled to full range (as distance mask are generated and not clamped)
        bin_thresholds=np.asarray([0.5, 1.5, 2.5]),
        tolerance_region=tolerance_region,
        downsampling_factor=downsampling_factor,
        mask_exclusion_zones=mask_exclusion_zones
    )


def prf1(downsampling_factor: int, mask_exclusion_zones: bool):
    return ccq(downsampling_factor, mask_exclusion_zones, tolerance_region=0.0)


def remove_detections_in_exclusion_zone(
    prediction: Dict[str, torch.Tensor], exclusion_zone: torch.Tensor, return_mask: bool = False
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    boxes = prediction["boxes"]
    x_center = torch.minimum(torch.round((boxes[:, 0] + boxes[:, 2]) / 2).int(), torch.tensor(4095))
    y_center = torch.minimum(torch.round((boxes[:, 1] + boxes[:, 3]) / 2).int(), torch.tensor(2999))
    outside_exclusion_zone = torch.logical_not(exclusion_zone[y_center, x_center])

    labels = prediction.get("labels", None)
    labels = labels[outside_exclusion_zone] if labels is not None else None
    scores = prediction.get("scores", None)
    scores = scores[outside_exclusion_zone] if scores is not None else None

    other_keys = set(prediction.keys()).difference({"boxes", "labels", "scores"})
    other_entities = {key: prediction[key] for key in other_keys}

    filtered_detections = {
        "boxes": boxes[outside_exclusion_zone],
        "labels": labels,
        "scores": scores,
        **other_entities
    }
    if return_mask:
        return filtered_detections, outside_exclusion_zone
    else:
        return filtered_detections
