import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Iterable, Callable, Optional

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from powerlines.data.utils import compute_extended_frame_padding


def load_yaml(filepath: Path) -> Dict[str, Any]:
    with filepath.open() as file:
        return yaml.safe_load(file)


def load_npy(filepath: Path) -> np.ndarray:
    return np.load(str(filepath))


def parallelize(
    function: Callable, data: Iterable, num_workers: int, description: Optional[str] = None, use_threads: bool = False
):
    concurrent_executor = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
    # Using executor, additionally with tqdm: https://stackoverflow.com/a/52242947
    with concurrent_executor(max_workers=num_workers) as executor:
        return list(tqdm(executor.map(function, data), total=len(list(data)), desc=description))


def pad_tensor_to_match_target_size(tensor: torch.Tensor, downsampling_factor: int, padding_value: float) -> torch.Tensor:
    pad_bottom, pad_right = compute_extended_frame_padding(tensor.shape, downsampling_factor)
    return torch.nn.functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode="constant", value=padding_value)


class MinPoolingDownsampler(nn.Module):
    def __init__(self, adjust_do_divisible: bool, factor: int):
        super().__init__()
        self._adjust_to_divisible = adjust_do_divisible
        self._factor = factor
        self._max_pooling = nn.MaxPool2d(
            kernel_size=(factor, factor),
            stride=(factor, factor)
        )

    def forward(self, distance_mask: torch.Tensor):
        max_value = distance_mask.max()
        inverted_distance_mask = max_value - distance_mask

        if self._adjust_to_divisible:
            input_tensor = pad_tensor_to_match_target_size(inverted_distance_mask, self._factor, padding_value=0)
        else:
            input_tensor = inverted_distance_mask

        pooled = self._max_pooling(input_tensor)
        return max_value - pooled
