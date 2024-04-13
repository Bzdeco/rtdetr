from pathlib import Path
from typing import Callable

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.dataset import TrainPolesDetectionDataset


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


def train_dataset(with_augmentations: bool = True) -> TrainPolesDetectionDataset:
    return TrainPolesDetectionDataset(
        data_source=data_source("train"),
        loading=loading(),
        sampling=sampling(),
        with_augmentations=with_augmentations,
        num_frames=10
    )


def val_dataset() -> None:
    raise NotImplementedError


def dataloader(
    dataset: Dataset,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    num_workers: int,
    collate_fn: Callable
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
