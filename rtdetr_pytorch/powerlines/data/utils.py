import math
import random
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple

import numpy as np

from powerlines.data.annotations import ImageAnnotations
from powerlines.data.config import DataSourceConfig, LoadingConfig
from powerlines.utils import load_yaml, load_npy

ROOT_FOLDER = Path(__file__).parents[1]
FOLDS = load_yaml(ROOT_FOLDER / "configs/powerlines/folds.yaml")
SPLITS = load_yaml(ROOT_FOLDER / "configs/powerlines/splits_timestamps.yaml")
INVALID_MASK_VALUE = np.iinfo(np.uint16).max


def sample_index_to_frame_id(num_cable_samples_per_frame: List[int], num_bg_samples_per_frame: int) -> Dict[int, int]:
    num_frames = len(num_cable_samples_per_frame)
    if num_frames == 0:
        return {}

    num_total_cable_samples = sum(num_cable_samples_per_frame)
    num_total_samples = num_total_cable_samples + num_frames * num_bg_samples_per_frame

    index_to_frame_id = {}
    current_frame_id = 0
    num_added_samples_to_current_frame = 0
    num_available_samples_in_curr_frame = num_cable_samples_per_frame[current_frame_id] + num_bg_samples_per_frame

    for idx in range(num_total_samples):
        while num_available_samples_in_curr_frame == 0:
            # Advance to next frame
            current_frame_id += 1
            num_added_samples_to_current_frame = 0
            num_available_samples_in_curr_frame = num_cable_samples_per_frame[current_frame_id] + num_bg_samples_per_frame

        if num_added_samples_to_current_frame >= num_available_samples_in_curr_frame:
            current_frame_id += 1
            num_added_samples_to_current_frame = 0
            num_available_samples_in_curr_frame = num_cable_samples_per_frame[current_frame_id] + num_bg_samples_per_frame

            while num_available_samples_in_curr_frame == 0:
                # Advance to next frame
                current_frame_id += 1
                num_added_samples_to_current_frame = 0
                num_available_samples_in_curr_frame = num_cable_samples_per_frame[current_frame_id] + num_bg_samples_per_frame

        index_to_frame_id[idx] = current_frame_id
        num_added_samples_to_current_frame += 1

    return index_to_frame_id


def num_background_samples_per_frame(
    num_cable_samples_per_frame: List[int],
    negative_sample_probability: float
):
    num_frames = len(num_cable_samples_per_frame)
    if num_frames == 0:
        return 0

    num_total_cable_samples = sum(num_cable_samples_per_frame)
    avg_num_cable_samples_per_frame = int(round(num_total_cable_samples / num_frames))
    positive_sample_probability = 1 - negative_sample_probability

    if positive_sample_probability > 0:
        return int(round(
            (negative_sample_probability * avg_num_cable_samples_per_frame) / positive_sample_probability
        ))
    else:
        return avg_num_cable_samples_per_frame


def num_pole_samples_in_frame(frame_annotation: ImageAnnotations) -> int:
    return len(frame_annotation.poles())


def load_filepaths(data_source_config) -> List[Path]:
    return sorted(list(filter(
        lambda path: data_source_config.belongs_to_subset_frames(path),
        data_source_config.inputs_folder.glob("*")
    )))


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


def load_filtered_filepaths(data_source_config) -> List[Path]:
    return _filter_excluded_filepaths(
        _filter_selected_filepaths(
            load_filepaths(data_source_config),
            data_source_config.selected_timestamps
        ), data_source_config.excluded_timestamps
    )


def filter_timestamps(timestamps: List[int], selected_timestamps: Set[int], excluded_timestamps: Set[int]) -> List[int]:
    return _filter_excluded_timestamps(_filter_selected_timestamps(timestamps, selected_timestamps), excluded_timestamps)


def _filter_excluded_filepaths(filepaths: List[Path], excluded_timestamps: Optional[Set[int]]) -> List[Path]:
    if excluded_timestamps is None:
        return filepaths

    return [filepath for filepath in filepaths if int(filepath.stem) not in excluded_timestamps]


def _filter_excluded_timestamps(timestamps: List[int], excluded_timestamps: Optional[Set[int]]) -> List[int]:
    if excluded_timestamps is None:
        return timestamps

    return [timestamp for timestamp in timestamps if timestamp not in excluded_timestamps]


def _filter_selected_filepaths(filepaths: List[Path], selected_timestamps: Optional[Set[int]]) -> List[Path]:
    if selected_timestamps is None:
        return filepaths

    return [filepath for filepath in filepaths if int(filepath.stem) in selected_timestamps]


def _filter_selected_timestamps(timestamps: List[int], selected_timestamps: Optional[Set[int]]) -> List[int]:
    if selected_timestamps is None:
        return timestamps

    return [timestamp for timestamp in timestamps if timestamp in selected_timestamps]


def train_fold_timestamps(val_fold: int, num_folds: int) -> List[int]:
    train_folds = sorted(list(set(range(num_folds)).difference({val_fold})))
    train_timestamps = []
    for fold in train_folds:
        train_timestamps.extend(FOLDS[f"fold_{fold}"])

    return train_timestamps


def val_fold_timestamps(val_fold: int) -> List[int]:
    return FOLDS[f"fold_{val_fold}"]


def split_timestamps(subset: str) -> List[int]:
    return SPLITS[subset]


def positive_sampled_patch_centers_data(
    frame_annotations: ImageAnnotations,
    distance_mask: np.ndarray,
    perturbation_size: int,
    patch_size: int,
    remove_excluded_area: bool
) -> Dict[str, np.ndarray]:
    sampling_within_distance_area = distance_mask[0, :, :] <= perturbation_size
    return _sampled_patch_centers(
        sampling_within_distance_area, patch_size, remove_excluded_area, frame_annotations
    )


def negative_sampled_patch_centers_data(
    frame_annotations: ImageAnnotations,
    distance_mask: np.ndarray,
    patch_size: int,
    remove_excluded_area: bool
) -> Dict[str, np.ndarray]:
    sampling_outside_distance_area = np.logical_and(
        distance_mask[0, :, :] > (patch_size / math.sqrt(2)),
        distance_mask[0, :, :] != INVALID_MASK_VALUE
    )
    return _sampled_patch_centers(
        sampling_outside_distance_area, patch_size, remove_excluded_area, frame_annotations
    )


def _sampled_patch_centers(
    sampling_area: np.ndarray,
    patch_size: int,
    remove_excluded_area: bool,
    frame_annotations: ImageAnnotations
) -> Dict[str, np.ndarray]:
    removed_margins = remove_image_margins(sampling_area, patch_size)

    if remove_excluded_area:
        final_sampling_area = remove_exclusion_area_from_mask(
            removed_margins, frame_annotations, perturbation_size=0, patch_size=patch_size
        )
    else:
        final_sampling_area = removed_margins

    patch_centers = patch_centers_from_sampling_area(final_sampling_area)
    ys = patch_centers[:, 0]
    non_sky_bias = int(sampling_area.shape[0] / 2)
    ground_region = (ys > non_sky_bias)
    sky_region = ~ground_region

    return {
        "has_ground_region": np.any(ground_region),
        "ground": patch_centers[ground_region],
        "sky": patch_centers[sky_region]
    }


def patch_centers_from_sampling_area(sampling_area: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(sampling_area)
    return np.stack((ys, xs), axis=1)


def remove_image_margins(sampling_area_mask: np.ndarray, patch_size: int) -> np.ndarray:
    inner_area = np.zeros_like(sampling_area_mask)
    half_patch_size = patch_size // 2
    inner_area[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size] = True
    return np.logical_and(sampling_area_mask, inner_area)


def remove_exclusion_area_from_mask(
    mask: np.ndarray, image_annotations: ImageAnnotations, perturbation_size: int, patch_size: int
) -> np.ndarray:
    exclusion_mask = exclusion_area_mask(image_annotations, perturbation_size, patch_size, mask.shape)
    return np.logical_and(mask, np.logical_not(exclusion_mask))


def exclusion_area_mask(
    image_annotations: ImageAnnotations, perturbation_size: int, patch_size: int, shape: Tuple[int, int]
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width))
    margin = perturbation_size + patch_size // 2

    for exclusion_zone in image_annotations.exclusion_zones:
        top, left = exclusion_zone.top_left
        bottom, right = exclusion_zone.bottom_right
        mask[
            max(0, int(top - margin)):min(height, int(bottom + margin + 1)),
            max(0, int(left - margin)):min(width, int(right + margin + 1))
        ] = True

    return mask


def sample_patch_center(
    patch_centers_data: Dict[str, np.ndarray], non_sky_bias: Optional[float] = None
) -> np.ndarray:
    if non_sky_bias is not None and random.random() <= non_sky_bias and patch_centers_data["has_ground_region"]:
        used_patch_centers = patch_centers_data["ground"]
    else:
        used_patch_centers = np.concatenate([patch_centers_data["sky"], patch_centers_data["ground"]])

    # Sample patch center
    patch_center_idx = random.randint(0, len(used_patch_centers) - 1)

    return used_patch_centers[patch_center_idx, :]
