from typing import Any, Dict

import torch
import torchvision.transforms.v2 as tv_transforms

from torchvision import tv_tensors


class RandomIoUCrop(tv_transforms.RandomIoUCrop):
    def __init__(
        self, min_scale: float, max_scale: float, min_aspect_ratio: float, max_aspect_ratio: float, trials: int, p: float
    ):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, trials=trials)
        self._p = p

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if torch.rand() < self._p:
            return super()._transform(inpt, params)
        return inpt


class NormalizeBoundingBoxes(tv_transforms.Transform):
    _transformed_types = (tv_tensors.BoundingBoxes,)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = inpt.spatial_size
        print("spatial_size", spatial_size)
        return inpt / spatial_size
