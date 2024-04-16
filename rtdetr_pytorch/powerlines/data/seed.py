# From https://pytorch.org/docs/stable/notes/randomness.html

import random
from typing import Optional

import numpy as np
import torch

SEED = 7


def set_global_seeds(seed: Optional[int] = None):
    seed = seed or SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_generator() -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
