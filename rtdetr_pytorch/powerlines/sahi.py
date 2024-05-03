import math
from collections import namedtuple
from typing import Dict, List, Callable, Optional

import torch
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from powerlines.data.utils import DETECTOR_INPUT_SIZE, cut_into_complete_set_of_patches
from powerlines.nms import nms, batched_nms

postprocess_bboxes = transforms.Compose([
    transforms.ClampBoundingBoxes(),
    transforms.ToPureTensor()
])


def merge_patch_boxes_predictions(
    predictions: List[Dict[str, torch.Tensor]], shifts: torch.Tensor, patch_sizes: torch.Tensor, orig_size: int
) -> Dict[str, torch.Tensor]:
    assert len(predictions) == len(shifts)
    scales = patch_sizes / orig_size
    device = predictions[0]["boxes"].device

    boxes = []
    for prediction, shift, scale in zip(predictions, shifts.to(device), scales.to(device)):
        xyxy_shift = shift.repeat(2).unsqueeze(0)
        boxes.append(scale * prediction["boxes"] + xyxy_shift)

    merged_boxes = BoundingBoxes(
        torch.concatenate(boxes, dim=0),
        format=BoundingBoxFormat.XYXY,
        canvas_size=(3000, 4096)
    )
    labels = torch.concatenate([pred["labels"] for pred in predictions], dim=0)
    scores = torch.concatenate([pred["scores"] for pred in predictions], dim=0)

    return postprocess_bboxes({"boxes": merged_boxes, "labels": labels, "scores": scores})


# Rows format: [x1, y1, x2, y2, score, label]
def vectorized_object_predictions(prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    return torch.concatenate((boxes, scores.unsqueeze(1), labels.unsqueeze(1)), dim=1)


def filter_valid_vect(vectorized_predictions: torch.Tensor) -> torch.Tensor:
    is_valid = torch.logical_and(
        vectorized_predictions[:, 0] < vectorized_predictions[:, 2],
        vectorized_predictions[:, 1] < vectorized_predictions[:, 3]
    )
    return vectorized_predictions[is_valid]


def filter_by_score_vect(vectorized_predictions: torch.Tensor, min_score: float) -> torch.Tensor:
    have_sufficient_score = vectorized_predictions[:, 4] >= min_score
    return vectorized_predictions[have_sufficient_score]


def vectorized_object_predictions_to_tensors(vectorized_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "boxes": vectorized_predictions[:, :4],
        "labels": vectorized_predictions[:, 5].long(),
        "scores": vectorized_predictions[:, 4]
    }


class VectorizedPostprocessPredictions:
    """
    Implementation from SAHI (https://github.com/obss/sahi) adopted to vectorized input
    Utilities for calculating IOU/IOS based match for given ObjectPredictions
    """

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

    def __call__(self, tensor: torch.Tensor):
        raise NotImplementedError()


class VectorizedNMSPostprocess(VectorizedPostprocessPredictions):
    """
    Implementation from SAHI (https://github.com/obss/sahi) adopted to vectorized input
    """
    def __call__(self, vectorized_predictions: torch.Tensor):
        if self.class_agnostic:
            keep = nms(
                vectorized_predictions, match_threshold=self.match_threshold, match_metric=self.match_metric
            )
        else:
            keep = batched_nms(
                vectorized_predictions, match_threshold=self.match_threshold, match_metric=self.match_metric
            )

        return vectorized_predictions[torch.as_tensor(keep).long()]


SAHI_POSTPROCESSOR = VectorizedNMSPostprocess()


def sahi_combine_predictions_to_full_resolution(
    patch_predictions: List[Dict[str, torch.Tensor]],
    shifts: torch.Tensor,
    patch_sizes: torch.Tensor,
    min_score: Optional[float] = None
) -> Dict[str, torch.Tensor]:
    merged_patch_predictions = merge_patch_boxes_predictions(patch_predictions, shifts, patch_sizes, DETECTOR_INPUT_SIZE[0])
    patch_predictions_vect = vectorized_object_predictions(merged_patch_predictions)
    patch_predictions_vect = filter_valid_vect(patch_predictions_vect)

    if min_score is not None:
        patch_predictions_vect = filter_by_score_vect(patch_predictions_vect, min_score)

    merged_vectorized_predictions = SAHI_POSTPROCESSOR(patch_predictions_vect)
    return vectorized_object_predictions_to_tensors(merged_vectorized_predictions)


MultiscalePatches = namedtuple("MultiscalePatches", ["patches", "shifts", "patch_sizes"])


def multiscale_image_patches(
    image: torch.Tensor, patch_sizes: List[int], step_size_fraction: float, predict_on_full_image: bool = False
):
    image_patches = []
    shifts = []
    sizes = []

    for patch_size in patch_sizes:
        step_size = int(step_size_fraction * patch_size)
        scale_patches, scale_shifts = cut_into_complete_set_of_patches(image.squeeze(), patch_size, step_size)
        image_patches.extend(scale_patches)
        shifts.extend(scale_shifts)
        sizes.extend([patch_size] * len(scale_patches))

    if predict_on_full_image:
        vertical_padding = 548  # = (4096 (width) - 3000 (height)) / 2
        padded_image = torch.nn.functional.pad(
            image.squeeze(0), (0, 0, vertical_padding, vertical_padding), mode="constant", value=0
        )
        image_patches.append(padded_image)
        shifts.append(torch.as_tensor([0, -vertical_padding]))
        sizes.append(4096)

    return MultiscalePatches(patches=image_patches, shifts=torch.stack(shifts), patch_sizes=torch.as_tensor(sizes))


def batch_multiscale_patches(
    multiscale_patches: MultiscalePatches, batch_size: int, preprocess: Callable
) -> List[torch.Tensor]:
    batches = []
    n_batches = int(math.ceil(len(multiscale_patches.patches) / batch_size))

    for b in range(n_batches):
        patches = multiscale_patches.patches[b * batch_size:(b + 1) * batch_size]
        batch = torch.stack([preprocess(patch) for patch in patches], dim=0)
        batches.append(batch)

    return batches
