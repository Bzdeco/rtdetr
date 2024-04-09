from pathlib import Path

from omegaconf import DictConfig

from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.dataset import PolesDetectionDataset


def data_source(subset: str) -> DataSourceConfig:
    return DataSourceConfig(
        complete_frames_root_folder=Path("/scratch/cvlab/home/gwizdala/dataset/processed/daedalean/complete_frames"),
        annotations_path=Path("/scratch/cvlab/home/gwizdala/dataset/daedalean/cable-annotations"),
        data_source_subset=subset,
        cv_config=DictConfig({"num_folds": 5, "fold": None, "folds_select": None, "parallel": []})
    )


def loading() -> LoadingConfig:
    return LoadingConfig(
        poles_distance_mask=True,
        exclusion_area_mask=False
    )


def sampling() -> SamplingConfig:
    return SamplingConfig(
        patch_size=1024,
        perturbation_size=384,
        negative_sample_prob=0.12,
        non_sky_bias=0.5,
        remove_excluded_area=True
    )


def train_dataset() -> PolesDetectionDataset:
    return PolesDetectionDataset(
        data_source=data_source("train"),
        loading=loading(),
        sampling=sampling(),
        num_frames=10  # FIXME: remove
    )


def val_dataset() -> None:
    raise NotImplementedError
