from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from powerlines.data.annotations import ImageAnnotations
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.utils import load_filtered_filepaths, load_annotations, load_complete_frame


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

        self.cache = [{
            "timestamp": annotation.frame_timestamp(),
            "annotation": annotation
        } for annotation in self.annotations.values()]

    def __getitem__(self, frame_id: int):
        frame = load_complete_frame(self.data_source, self.loading, self.cache[frame_id])
        annotation = frame["annotation"]

        input = tv_tensors.Image(torch.from_numpy(frame["image"]))
        bounding_boxes = self._bounding_boxes(annotation, "poles")
        poles_distance_mask = frame.get("poles_distance_mask", None),
        exclusion_zones_distance_mask = frame.get("exclusion_zones_distance_mask", None)
        if exclusion_zones_distance_mask is not None:
            exclusion_zone = (exclusion_zones_distance_mask == 0).squeeze()
            boxes_exclusion_zone = self._bounding_boxes(annotation, "exclusion_zone")
        else:
            exclusion_zone = None
            boxes_exclusion_zone = None

        targets = {
            "boxes": bounding_boxes,
            "labels": torch.as_tensor([0] * len(bounding_boxes)).long(),
            "poles_distance_mask": torch.from_numpy(poles_distance_mask),
            "exclusion_zone": exclusion_zone,
            "boxes_exclusion_zone": boxes_exclusion_zone
        }

        return input, targets

    def _bounding_boxes(self, annotation: ImageAnnotations, entity: str = "poles") -> tv_tensors.BoundingBoxes:
        bounding_boxes = []
        annotated_boxes = annotation.poles() if entity == "poles" else annotation.exclusion_zones

        for box in annotated_boxes:
            y_0, x_0 = box.top_left
            y_1, x_1 = box.bottom_right
            bounding_boxes.append([x_0, y_0, x_1, y_1])

        return tv_tensors.BoundingBoxes(
            bounding_boxes if len(bounding_boxes) > 0 else torch.empty((0, 4)),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(3000, 4096)
        )

    def __len__(self) -> int:
        return self.num_frames
