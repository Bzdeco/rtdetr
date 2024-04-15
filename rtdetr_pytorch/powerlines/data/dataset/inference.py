from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from powerlines.data.annotations import ImageAnnotations
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.utils import evaluation_augmentations, load_filtered_filepaths, load_annotations, load_complete_frame


ORIG_SIZE = torch.tensor([3000, 4096])


class InferencePolesDetectionDataset(Dataset):
    def __init__(
        self,
        data_source: DataSourceConfig,
        loading: LoadingConfig,
        sampling: SamplingConfig,
        num_frames: Optional[int] = None
    ):
        self.data_source = data_source
        self.loading = loading
        self.sampling = sampling

        self.filepaths = load_filtered_filepaths(data_source)
        self.annotations = load_annotations(data_source)
        self.num_frames = num_frames if num_frames is not None else len(self.filepaths)

        self.augmentations = evaluation_augmentations()
        self.cache = [{
            "timestamp": annotation.frame_timestamp(),
            "annotation": annotation
        } for annotation in self.annotations.values()]

    def __getitem__(self, frame_id: int):
        frame = load_complete_frame(self.data_source, self.loading, self.cache[frame_id])
        input = tv_tensors.Image(torch.from_numpy(frame["image"]))
        bounding_boxes = self._bounding_boxes(frame["annotation"])
        poles_distance_mask = frame.get("poles_distance_mask", None)

        targets = {
            "boxes": bounding_boxes,
            "labels": torch.as_tensor([0] * len(bounding_boxes))
        }
        input_aug, targets_aug = self.augmentations(input, targets)
        targets_aug["poles_distance_mask"] = torch.from_numpy(poles_distance_mask)
        targets_aug["labels"] = targets_aug["labels"].long()
        targets_aug["orig_size"] = ORIG_SIZE

        return input_aug, targets_aug

    def _bounding_boxes(self, annotation: ImageAnnotations) -> tv_tensors.BoundingBoxes:
        bounding_boxes = []
        for pole in annotation.poles():
            y_0, x_0 = pole.top_left
            y_1, x_1 = pole.bottom_right
            bounding_boxes.append([x_0, y_0, x_1, y_1])

        return tv_tensors.BoundingBoxes(
            bounding_boxes if len(bounding_boxes) > 0 else torch.empty((0, 4)),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(3000, 4096)
        )

    def __len__(self) -> int:
        return self.num_frames
