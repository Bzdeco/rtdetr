from typing import Dict, List

import torch
from sahi import ObjectPrediction
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


def tensors_to_sahi_object_predictions(prediction: Dict[str, torch.Tensor]) -> List[ObjectPrediction]:
    object_predictions = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        object_predictions.append(ObjectPrediction(
            bbox=box.tolist(),
            category_id=label,
            category_name="pole",
            score=score,
            full_shape=[3000, 4096]
        ))
    return object_predictions


def sahi_object_predictions_to_tensors(object_predictions: List[ObjectPrediction]) -> Dict[str, torch.Tensor]:
    boxes, labels, scores = [], [], []
    for object_prediction in object_predictions:
        boxes.append(torch.as_tensor(object_prediction.bbox))
        labels.append(object_prediction.category.id)
        scores.append(object_prediction.score)

    return {"boxes": torch.stack(boxes, dim=0), "labels": torch.as_tensor(labels), "scores": torch.as_tensor(scores)}
