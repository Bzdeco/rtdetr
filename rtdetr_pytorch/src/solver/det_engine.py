"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import sys
from typing import Iterable, Callable

import torch
import torch.amp
from neptune import Run
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from powerlines.data.utils import inference_augmentations, ORIG_SIZE
from powerlines.evaluation import mean_average_precision, remove_detections_in_exclusion_zone
from powerlines.sahi import sahi_combine_predictions_to_full_resolution, multiscale_image_patches, batch_multiscale_patches
from powerlines.visualization import VisualizationLogger
from src.misc import (MetricsTracker, reduce_dict)


def train_one_epoch(config: DictConfig, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metrics_tracker = MetricsTracker()

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in tqdm(data_loader, desc="Training"):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # EMA
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metrics_tracker.update(loss=loss_value, **loss_dict_reduced)
        metrics_tracker.update(lr=optimizer.param_groups[0]["lr"])

    return {name: meter.global_avg for name, meter in metrics_tracker.meters.items()}


@torch.no_grad()
def evaluate(
    epoch: int,
    config: DictConfig,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    detection_postprocessor: Callable,
    data_loader: DataLoader,
    device: torch.device,
    run: Run
):
    model.eval()
    criterion.eval()

    # Create mAP metrics
    map_all = mean_average_precision()
    map_exclusion_zones = mean_average_precision()

    logger = VisualizationLogger(run, config)

    preprocess = inference_augmentations()
    for image, target in tqdm(data_loader, desc="Validating"):
        image = image.to(device)
        target = {k: v.to(device) for k, v in target[0].items()}

        sahi_config = config.sahi
        multiscale_patches = multiscale_image_patches(
            image,
            patch_sizes=sahi_config.patch_sizes,
            step_size_fraction=sahi_config.step_size_fraction
        )

        patch_predictions = []
        with torch.autocast(device_type=str(device)):
            for batch in batch_multiscale_patches(
                multiscale_patches, batch_size=sahi_config.batch_size, preprocess=preprocess
            ):
                batch_outputs = model(batch)
                patch_predictions.extend(detection_postprocessor(
                    batch_outputs, torch.stack([ORIG_SIZE] * len(batch), dim=0).to(device)
                ))

        prediction = sahi_combine_predictions_to_full_resolution(
            patch_predictions,
            multiscale_patches.shifts,
            multiscale_patches.patch_sizes,
            device,
            min_score=sahi_config.min_score
        )

        # Consider exclusion zones in predictions and targets, if present
        exclusion_zone = target["exclusion_zone"]
        prediction_excl_zones = remove_detections_in_exclusion_zone(prediction, exclusion_zone)
        target_excl_zones = remove_detections_in_exclusion_zone(target, exclusion_zone)
        _ = map_exclusion_zones([prediction_excl_zones], [target_excl_zones])
        logger.log(epoch, image, prediction_excl_zones, target_excl_zones)

        # Metrics without exclusion zones
        _ = map_all([prediction], [target])

    return {"metrics/all": map_all.compute(), "metrics/masked": map_exclusion_zones.compute()}
