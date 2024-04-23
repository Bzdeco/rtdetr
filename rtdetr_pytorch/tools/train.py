"""by lyuwenyu and Bzdeco
"""
from typing import Optional

from powerlines.data.seed import set_global_seeds
set_global_seeds()

from hydra import initialize, compose
from omegaconf import DictConfig
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import src.misc.dist as dist
from src.core import YAMLConfig 
from src.solver import DetSolver


def rt_detr_config() -> YAMLConfig:
    return YAMLConfig(
        "configs/rtdetr/rtdetr_r18vd_6x_coco.yml",
        resume="",
        use_amp=True,
        tuning="checkpoints/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
    )


def powerlines_config() -> DictConfig:
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="powerlines")


def run_training(cfg_powerlines: DictConfig) -> Optional[float]:  # optimized metric value
    dist.init_distributed()

    # Create and configure solver
    cfg = rt_detr_config()
    solver = DetSolver(cfg, cfg_powerlines)
    return solver.fit()


if __name__ == '__main__':
    run_training(powerlines_config())
