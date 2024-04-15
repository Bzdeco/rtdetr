import random

from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from powerlines.data.annotations import ImageAnnotations
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.utils import load_filtered_filepaths, sample_patch_center, load_annotations, train_augmentations, \
    evaluation_augmentations, load_parameters_for_sampling, load_complete_frame, ORIG_SIZE, MAX_DISTANCE_MASK_VALUE
from powerlines.utils import parallelize


class TrainPolesDetectionDataset(Dataset):
    def __init__(
        self,
        data_source: DataSourceConfig,
        loading: LoadingConfig,
        sampling: SamplingConfig,
        num_frames: Optional[int] = None,
        with_augmentations: bool = True,
        num_workers: int = 4
    ):
        self.data_source = data_source
        self.loading = loading
        self.sampling = sampling
        self.with_augmentations = with_augmentations

        self.filepaths = load_filtered_filepaths(data_source)
        self.annotations = load_annotations(data_source)
        self.num_frames = num_frames if num_frames is not None else len(self.filepaths)

        if self.with_augmentations:
            self.augmentations = train_augmentations()
        else:
            self.augmentations = evaluation_augmentations()

        self._loading_data = self._frames_loading_data()
        self.cache = parallelize(
            load_parameters_for_sampling,
            self._loading_data,
            num_workers,
            f"Loading {data_source.data_source_subset} frames for configuration",
            use_threads=True
        )
        self.sampling.configure_sampling(self.cache)

    def _frames_loading_data(self) -> List[Dict[str, Any]]:
        return [self._single_frame_loading_data(frame_id) for frame_id in range(self.num_frames)]

    def _single_frame_loading_data(self, frame_id: int) -> Dict[str, Any]:
        filepath = self.filepaths[frame_id]
        timestamp = int(filepath.stem)
        return {
            "data_source": self.data_source,
            "loading": self.loading,
            "sampling": self.sampling,
            "timestamp": timestamp,
            "annotation": self.annotations[timestamp],
        }

    def __getitem__(self, idx: int):
        frame_id = self.sampling.frame_idx_for_sample(idx)
        frame = load_complete_frame(self.data_source, self.loading, self.cache[frame_id])

        if self._should_sample_positive_sample(frame):
            patch_centers_data = frame["positive_sampling_centers_data"]
        else:
            patch_centers_data = frame["negative_sampling_centers_data"]

        y, x = sample_patch_center(patch_centers_data, self.sampling.non_sky_bias)
        input = tv_tensors.Image(torch.from_numpy(self._extract_patch(frame["image"], y, x)))
        bounding_boxes = self._extract_bounding_boxes(frame["annotation"], y, x)
        poles_distance_mask_patch = self._poles_distance_mask_patch(frame, y, x)

        targets = {
            "boxes": bounding_boxes,
            "labels": torch.as_tensor([0] * len(bounding_boxes)),
        }

        input_aug, targets_aug = self.augmentations(input, targets)
        targets_aug["poles_distance_mask"] = torch.from_numpy(poles_distance_mask_patch)
        targets_aug["labels"] = targets_aug["labels"].long()
        if not self.with_augmentations:
            targets_aug["orig_size"] = ORIG_SIZE

        return input_aug, targets_aug

    def _extract_patch(self, input: Optional[np.ndarray], y: int, x: int) -> Optional[np.ndarray]:
        if input is None:
            return None

        return input[
            ...,
            (y - self.sampling.half_patch_size):(y + self.sampling.half_patch_size),
            (x - self.sampling.half_patch_size):(x + self.sampling.half_patch_size)
        ]

    def _poles_distance_mask_patch(self, cached_frame: Dict[str, Any], y: int, x: int) -> Optional[np.ndarray]:
        poles_distance_mask = self._extract_patch(cached_frame["poles_distance_mask"], y, x)
        if poles_distance_mask is None:
            return None

        if np.all(poles_distance_mask > 0):
            return np.full(
                (1, self.sampling.patch_size, self.sampling.patch_size),
                fill_value=MAX_DISTANCE_MASK_VALUE
            )
        else:
            return poles_distance_mask

    def _extract_bounding_boxes(self, annotation: ImageAnnotations, y: int, x: int) -> tv_tensors.BoundingBoxes:
        hps = self.sampling.half_patch_size
        patch_x_0 = x - hps
        patch_y_0 = y - hps

        bounding_boxes = []
        for pole in annotation.poles():
            x_pole, y_pole = pole.center_xy()
            if patch_x_0 <= x_pole < x + hps and patch_y_0 <= y_pole < y + hps:
                y_0, x_0 = pole.top_left
                y_1, x_1 = pole.bottom_right
                bounding_boxes.append([x_0 - patch_x_0, y_0 - patch_y_0, x_1 - patch_x_0, y_1 - patch_y_0])

        return tv_tensors.BoundingBoxes(
            bounding_boxes if len(bounding_boxes) > 0 else torch.empty((0, 4)),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(self.sampling.patch_size,) * 2
        )

    def _should_sample_positive_sample(self, cached_frame: Dict[str, Any]):
        should_sample_positive = random.random() <= self.sampling.positive_sample_prob
        has_positive_samples = cached_frame["has_positive_samples"]
        has_negative_samples = cached_frame["has_negative_samples"]

        return (should_sample_positive and has_positive_samples) or not has_negative_samples

    def __len__(self):
        return self.sampling.num_samples
