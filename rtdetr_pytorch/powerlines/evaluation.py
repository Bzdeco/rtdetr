from typing import Dict

import torch
from torchmetrics.detection import MeanAveragePrecision


def mean_average_precision():
    # Must be used together with the postprocessor
    return MeanAveragePrecision(
        box_format="xyxy",
        average="macro",
        backend="faster_coco_eval"
    )


def remove_detections_in_exclusion_zone(
    prediction: Dict[str, torch.Tensor], exclusion_zone: torch.Tensor
) -> Dict[str, torch.Tensor]:
    boxes = prediction["boxes"]
    x_center = torch.round((boxes[:, 0] + boxes[:, 2]) / 2).int()
    y_center = torch.round((boxes[:, 1] + boxes[:, 3]) / 2).int()
    outside_exclusion_zone = torch.logical_not(exclusion_zone[y_center, x_center])

    labels = prediction.get("labels", None)
    labels = labels[outside_exclusion_zone] if labels is not None else None
    scores = prediction.get("scores", None)
    scores = scores[outside_exclusion_zone] if scores is not None else None

    other_keys = set(prediction.keys()).difference({"boxes", "labels", "scores"})
    other_entities = {prediction[key] for key in other_keys}

    return {
        "boxes": boxes[outside_exclusion_zone],
        "labels": labels,
        "scores": scores,
        **other_entities
    }
