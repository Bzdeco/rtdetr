from typing import Dict, List

import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


def merge_patch_boxes_predictions(
    predictions: List[Dict[str, torch.Tensor]], shifts: torch.Tensor, patch_size: int, orig_size: int
):
    assert len(predictions) == len(shifts)

    boxes = []
    scale = patch_size / orig_size
    for prediction, shift in zip(predictions, shifts):
        xyxy_shift = shift.repeat(2)
        boxes.append(scale * prediction["boxes"] + xyxy_shift)

    merged_boxes = BoundingBoxes(
        torch.stack(boxes),
        format=BoundingBoxFormat.XYXY,
        canvas_size=(patch_size,) * 2
    )
    labels = torch.concatenate([pred["labels"] for pred in predictions], dim=0)
    scores = torch.concatenate([pred["scores"] for pred in predictions], dim=0)

    return {"boxes": merged_boxes, "labels": labels, "scores": scores}
