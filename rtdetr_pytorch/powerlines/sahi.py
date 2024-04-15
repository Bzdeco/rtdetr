from typing import Dict, List

import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


def merge_patch_boxes_predictions(
    predictions: List[Dict[str, torch.Tensor]], shifts: torch.Tensor, patch_size: int, orig_size: int
) -> Dict[str, torch.Tensor]:
    assert len(predictions) == len(shifts)
    scale = patch_size / orig_size
    device = predictions[0]["boxes"].device

    boxes = []
    for prediction, shift in zip(predictions, shifts.to(device)):
        xyxy_shift = shift.repeat(2)
        boxes.append(scale * prediction["boxes"] + xyxy_shift.unsqueeze(0))

    merged_boxes = BoundingBoxes(
        torch.concatenate(boxes, dim=0),
        format=BoundingBoxFormat.XYXY,
        canvas_size=(patch_size,) * 2
    )
    labels = torch.concatenate([pred["labels"] for pred in predictions], dim=0)
    scores = torch.concatenate([pred["scores"] for pred in predictions], dim=0)

    return {"boxes": merged_boxes.as_subclass(torch.Tensor), "labels": labels, "scores": scores}
