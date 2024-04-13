import random

from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode

from powerlines.data.annotations import ImageAnnotations
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.utils import load_filtered_filepaths, num_pole_samples_in_frame, positive_sampled_patch_centers_data, \
    negative_sampled_patch_centers_data, sample_patch_center
from powerlines.utils import parallelize, load_npy


MAX_DISTANCE_MASK_VALUE = 128


def load_annotations(data_source_config: DataSourceConfig) -> Dict[int, Any]:
    fold_annotations = {}

    for frame_annotations in data_source_config.annotations():
        frame_timestamp = frame_annotations.frame_timestamp()
        if frame_timestamp in data_source_config.timestamps:
            fold_annotations[frame_timestamp] = frame_annotations

    return fold_annotations


def load_distance_masks(
    data_source_config: DataSourceConfig, loading_config: LoadingConfig, frame_timestamp: int
) -> Dict[str, Optional[np.ndarray]]:
    frame_poles_distance_mask = load_npy(
        data_source_config.poles_distance_masks_folder / f"{frame_timestamp}.npy"
    ) if loading_config.poles_distance_mask else None

    return {
        "poles_distance_mask": frame_poles_distance_mask
    }


def load_parameters_for_configuration(loaded_frame_data: Dict[str, Any]) -> Dict[str, Any]:
    data_source: DataSourceConfig = loaded_frame_data["data_source"]
    sampling: SamplingConfig = loaded_frame_data["sampling"]
    loading: LoadingConfig = loaded_frame_data["loading"]
    timestamp = loaded_frame_data["timestamp"]
    annotation = loaded_frame_data["annotation"]
    perturbation_size = sampling.perturbation_size
    patch_size = sampling.patch_size
    remove_excluded_area = sampling.remove_excluded_area
    distance_masks = load_distance_masks(data_source, loading, timestamp)

    # Sample relative to poles
    sampling_source_distance_mask = distance_masks["poles_distance_mask"]
    num_positive_samples = num_pole_samples_in_frame(annotation)

    positive_patch_centers_data = positive_sampled_patch_centers_data(
        annotation,
        sampling_source_distance_mask,
        perturbation_size,
        patch_size,
        remove_excluded_area
    )
    negative_patch_centers_data = negative_sampled_patch_centers_data(
        annotation,
        sampling_source_distance_mask,
        patch_size,
        remove_excluded_area
    )

    return {
        "timestamp": timestamp,
        "annotation": annotation,
        "positive_sampling_centers_data": positive_patch_centers_data,
        "negative_sampling_centers_data": negative_patch_centers_data,
        "has_positive_samples": len(positive_patch_centers_data["ground"]) + len(positive_patch_centers_data["sky"]) > 0,
        "has_negative_samples": len(negative_patch_centers_data["ground"]) + len(negative_patch_centers_data["sky"]) > 0,
        "num_positive_samples": num_positive_samples
    }


def load_complete_frame(
    data_source: DataSourceConfig,
    loading: LoadingConfig,
    loaded_frame_data: Dict[str, Any]
) -> Dict[str, Any]:
    timestamp = loaded_frame_data["timestamp"]
    frame_image = load_npy(data_source.input_filepath(timestamp))
    distance_masks = load_distance_masks(data_source, loading, timestamp)

    return {
        "timestamp": timestamp,
        "image": frame_image,
        "poles_distance_mask": distance_masks["poles_distance_mask"],
        **loaded_frame_data
    }


DETECTOR_INPUT_SIZE = [640, 640]
MIN_BBOX_SIZE = 1  # min pole width across DDLN dataset = 0.5


def train_augmentations():
    # Augmentations from RT-DETR, without ZoomOut, with adjusted min_scale and min_size, and with fixed aspect ratio
    return transforms.Compose([
        transforms.RandomPhotometricDistort(
            brightness=(0.875, 1.125), contrast=(0.5, 1.5), hue=(-0.05, 0.05), saturation=(0.5, 1.5), p=0.5
        ),
        transforms.RandomIoUCrop(min_scale=1/8, max_scale=1, min_aspect_ratio=1, max_aspect_ratio=1, trials=40),
        transforms.SanitizeBoundingBoxes(MIN_BBOX_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(size=DETECTOR_INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.SanitizeBoundingBoxes(MIN_BBOX_SIZE),
        transforms.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.CXCYWH)
    ])


def evaluation_augmentations():
    # Only resizes input and adapts the bounding boxes format
    return transforms.Compose([
        transforms.Resize(size=DETECTOR_INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.SanitizeBoundingBoxes(MIN_BBOX_SIZE),
        transforms.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.CXCYWH)
    ])


class TrainPolesDetectionDataset(Dataset):
    def __init__(
        self,
        data_source: DataSourceConfig,
        loading: LoadingConfig,
        sampling: SamplingConfig,
        num_frames: Optional[int] = None,
        with_augmentations: bool = True,
        num_workers: int = 16
    ):
        self.data_source = data_source
        self.loading = loading
        self.sampling = sampling

        self.filepaths = load_filtered_filepaths(data_source)
        self.annotations = load_annotations(data_source)
        self.num_frames = num_frames if num_frames is not None else len(self.filepaths)

        if with_augmentations:
            self.augmentations = train_augmentations()
        else:
            self.augmentations = evaluation_augmentations()

        self._loading_data = self._frames_loading_data()
        self.cache = parallelize(
            load_parameters_for_configuration,
            self._loading_data,
            num_workers,
            f"Loading {data_source.data_source_subset} frames for configuration"
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
