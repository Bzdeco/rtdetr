import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Iterable, Callable, Optional

import numpy as np
import yaml
from tqdm import tqdm


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
