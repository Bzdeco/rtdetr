from typing import Dict, Tuple, Union

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from powerlines.ccq import CCQMetric


def mean_average_precision():
    # Must be used together with the postprocessor
    return MeanAveragePrecision(
        box_format="xyxy",
        average="macro",
        backend="faster_coco_eval"
    )


def ccq(mask_exclusion_zones: bool):
    return CCQMetric(
        bin_thresholds=np.asarray([0.0039, 0.01, 0.05, 0.1, 0.15, 0.1616, 0.2]),  # standard + default + NEVBW 2 cells distance
        tolerance_region=1.42,
        mask_exclusion_zones=mask_exclusion_zones
    )


def remove_detections_in_exclusion_zone(
    prediction: Dict[str, torch.Tensor], exclusion_zone: torch.Tensor, return_mask: bool = False
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    boxes = prediction["boxes"]
    x_center = torch.round((boxes[:, 0] + boxes[:, 2]) / 2).int()
    y_center = torch.round((boxes[:, 1] + boxes[:, 3]) / 2).int()
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
