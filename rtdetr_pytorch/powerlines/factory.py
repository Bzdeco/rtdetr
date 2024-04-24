import re
from pathlib import Path
from typing import List, Dict, Any

import torch.optim
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

from powerlines.data import seed
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.dataset.sampling import TrainPolesDetectionDataset
from powerlines.data.dataset.inference import InferencePolesDetectionDataset
from src.data.dataloader import default_collate_fn


def data_source(config: DictConfig, subset: str) -> DataSourceConfig:
    paths_config = config.paths
    return DataSourceConfig(
        complete_frames_root_folder=Path(paths_config.complete_frames),
        annotations_path=Path(paths_config.annotations),
        data_source_subset=subset,
        cv_config=config.data.cv
    )


def loading(config: DictConfig) -> LoadingConfig:
    return LoadingConfig(
        poles_distance_mask=True,
        exclusion_area_mask=True
    )


def sampling(config: DictConfig) -> SamplingConfig:
    data_config = config.data
    return SamplingConfig(
        patch_size=data_config.patch_size,
        perturbation_size=data_config.perturbation,
        negative_sample_prob=data_config.negative_sample_prob,
        non_sky_bias=data_config.non_sky_bias,
        remove_excluded_area=True
    )


def train_dataset(config: DictConfig, with_augmentations: bool = True) -> TrainPolesDetectionDataset:
    return TrainPolesDetectionDataset(
        config=config,
        data_source=data_source(config, "train"),
        loading=loading(config),
        sampling=sampling(config),
        with_augmentations=with_augmentations,
        num_frames=config.data.size.train
    )


def val_dataset(config: DictConfig) -> InferencePolesDetectionDataset:
    return InferencePolesDetectionDataset(
        data_source=data_source(config, "val"),
        loading=loading(config),
        sampling=sampling(config),
        num_frames=config.data.size.val
    )


def dataloader(
    dataset: Dataset,
    batch_size: int,
    drop_last: bool,
    shuffle: bool,
    num_workers: int,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
        generator=seed.torch_generator(),
        worker_init_fn=seed.seed_worker
    )


def optimizer(config: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_config = config.optimizer

    params_config = list(map(lambda param_group_config: dict(param_group_config), optimizer_config.params))
    return torch.optim.AdamW(
        params=get_optimizer_parameter_groups(params_config, model),
        lr=optimizer_config.lr,
        betas=tuple(optimizer_config.betas),
        weight_decay=optimizer_config.wd
    )


def get_optimizer_parameter_groups(params_config_list: List[Dict[str, Any]], model: nn.Module) -> List[Dict[str, Any]]:
    """
    Adapted from `YAMLConfig.get_optim_params`.

    E.g.:
        ^(?=.*a)(?=.*b).*$         means including a and b
        ^((?!b.)*a((?!b).)*$       means including a but not b
        ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
    """

    param_groups = []
    visited = []
    for pg in params_config_list:
        pattern = pg["params"]
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
        pg["params"] = list(params.values())
        param_groups.append(pg)
        visited.extend(list(params.keys()))

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
        param_groups.append({"params": list(params.values())})
        visited.extend(list(params.keys()))

    assert len(visited) == len(names), "Haven't visited all named parameters"
    return param_groups


def lr_scheduler(config: DictConfig, opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    milestones = config.lr_scheduler.milestones
    batch_size = config.data.batch_size
    # 16 was default batch size for the set milestone, adapt to larger or smaller batch size (i.e. more examples seen)
    scaled_milestones = [int(milestone * (batch_size / 16)) for milestone in milestones]

    return MultiStepLR(
        optimizer=opt,
        milestones=scaled_milestones,
        gamma=config.lr_scheduler.gamma
    )
