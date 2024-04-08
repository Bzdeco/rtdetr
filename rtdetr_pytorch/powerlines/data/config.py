from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Set

from omegaconf import DictConfig

from powerlines.data.annotations import parse_annotations
from powerlines.data.utils import filter_timestamps, train_fold_timestamps, val_fold_timestamps, split_timestamps, \
    num_background_samples_per_frame


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

    def input_filepath(self, timestamp: str):
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
