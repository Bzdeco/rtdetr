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
from torch.utils.data import DataLoader
from tqdm import tqdm

from powerlines.data.utils import cut_into_complete_set_of_patches, inference_augmentations, ORIG_SIZE
from powerlines.evaluation import mean_average_precision
from powerlines.sahi import sahi_sliced_predictions_to_full_resolution
from powerlines.visualization import VisualizationLogger
from src.misc import (MetricsTracker, reduce_dict)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
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
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    detection_postprocessor: Callable,
    data_loader: DataLoader,
    device: torch.device,
    run: Run
):
    model.eval()
    criterion.eval()

    # Create mAP metric
    mAP = mean_average_precision()
    logger = VisualizationLogger(run, n_images_per_epoch=3, every=1)

    preprocess = inference_augmentations()
    patch_size = 1024
    step_size = 512

    for image, target in tqdm(data_loader, desc="Validating"):
        image = image.to(device)
        target = {k: v.to(device) for k, v in target[0].items()}

        image_patches, shifts = cut_into_complete_set_of_patches(image.squeeze(), patch_size, step_size)
        with torch.autocast(device_type=str(device)):
            input = preprocess(image_patches)
            patch_outputs = model(input)  # assumes this batch size will fit

        patch_predictions = detection_postprocessor(
            patch_outputs, torch.stack([ORIG_SIZE] * len(image_patches), dim=0).to(device)
        )
        prediction = sahi_sliced_predictions_to_full_resolution(patch_predictions, shifts, patch_size, device)

        _ = mAP([prediction], [target])
        logger.log(epoch, image, prediction, target)

    return mAP.compute()
