from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Set

from omegaconf import DictConfig

from powerlines.data.annotations import parse_annotations, ImageAnnotations
from powerlines.utils import load_yaml


@dataclass
class DataSourceConfig:
    complete_frames_root_folder: Path
    annotations_path: Path
    data_source_subset: str
    cv_config: DictConfig
    selected_timestamps: Optional[Set[int]] = field(default=None, repr=False)
    excluded_timestamps: Optional[Set[int]] = field(default=None, repr=False)

    inputs_folder: Path = field(init=False)
    poles_distance_masks_folder: Path = field(init=False)
    exclusion_zones_distance_masks_folder: Path = field(init=False)

    def __post_init__(self):
        self.inputs_folder = self.complete_frames_root_folder / "images"
        self.poles_distance_masks_folder = self.complete_frames_root_folder / "poles_distance_masks"
        self.exclusion_zones_distance_masks_folder = self.complete_frames_root_folder / "exclusion_zones_distance_masks"

        if self.cv_config.fold is None:  # fixed train-val split
            if self.data_source_subset is None:
                # Use all available timestamps
                self.timestamps = sorted(list(map(lambda path: int(path.stem), self.inputs_folder.glob("*.npy"))))
            else:
                self.timestamps = set(filter_timestamps(
                    split_timestamps(self.data_source_subset), self.selected_timestamps, self.excluded_timestamps
                ))
        else:  # fold-based dataset split
            if self.data_source_subset == "train":
                timestamps = train_fold_timestamps(self.cv_config.fold, self.cv_config.num_folds)
            else:
                timestamps = val_fold_timestamps(self.cv_config.fold)
            self.timestamps = set(filter_timestamps(
                timestamps, self.selected_timestamps, self.excluded_timestamps
            ))

    def annotations(self):
        return parse_annotations(self.annotations_path)

    def input_filepath(self, timestamp: int):
        return self.inputs_folder / f"{timestamp}.npy"

    def subsets_split(self) -> bool:
        return self.cv_config.fold is None

    def folds_split(self) -> bool:
        return self.cv_config.fold is not None

    def belongs_to_subset_frames(self, path: Path) -> bool:
        timestamp = int(path.stem)
        return timestamp in self.timestamps


@dataclass
class LoadingConfig:
    poles_distance_mask: bool
    exclusion_area_mask: bool


@dataclass
class SamplingConfig:
    patch_size: int
    perturbation_size: int
    negative_sample_prob: Optional[float] = None
    non_sky_bias: Optional[float] = None
    remove_excluded_area: bool = True

    half_patch_size: int = field(init=False)
    positive_sample_prob: float = field(init=False)
    num_samples: int = field(init=False)
    num_neg_samples_per_frame: int = field(init=False)

    _sample_index_to_frame_id: Dict[int, int] = field(init=False)

    def __post_init__(self):
        self.half_patch_size = self.patch_size // 2
        self.positive_sample_prob = 1 - self.negative_sample_prob

    def configure_sampling(self, cache: List[Dict[str, Any]]):
        # Sampling indices mapping to frames
        num_positive_samples_per_frame = list(map(
            lambda frame_cache: frame_cache["num_positive_samples"] if frame_cache["has_positive_samples"] else 0,
            cache
        ))
        self.num_neg_samples_per_frame = num_background_samples_per_frame(
            num_positive_samples_per_frame, self.negative_sample_prob
        )
        self._sample_index_to_frame_id = sample_index_to_frame_id(
            num_positive_samples_per_frame, self.num_neg_samples_per_frame
        )
        self.num_samples = len(self._sample_index_to_frame_id)

    def frame_idx_for_sample(self, sample_idx: int) -> int:
        return self._sample_index_to_frame_id[sample_idx]


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


ROOT_FOLDER = Path(__file__).parents[2]
FOLDS = load_yaml(ROOT_FOLDER / "configs/powerlines/folds.yaml")
SPLITS = load_yaml(ROOT_FOLDER / "configs/powerlines/splits_timestamps.yaml")


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
