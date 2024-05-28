"""by lyuwenyu and Bzdeco
"""
import argparse
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

    solver = DetSolver(rt_detr_config(), cfg_powerlines)
    return solver.fit()


def run_validation(run_id: int, cfg_powerlines: DictConfig) -> None:
    dist.init_distributed()

    print(f"Validating run {run_id}")
    for epoch in range(cfg_powerlines.validate_epochs):
        print(f"e={epoch}")
        cfg_powerlines.checkpoint.resume = True
        cfg_powerlines.checkpoint.run_id = run_id
        cfg_powerlines.checkpoint.epoch = epoch

        solver = DetSolver(rt_detr_config(), cfg_powerlines)
        solver.val(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=False, type=int)
    parser.add_argument("--resume", required=False, type=int)
    parser.add_argument("--resume_epoch", required=False, type=int)
    parser.add_argument("--validate_run", required=False, type=int)
    args = parser.parse_args()

    powerlines_cfg = powerlines_config()

    if args.fold is not None:
        fold = int(args.fold)
        powerlines_cfg.cv_name = powerlines_cfg.name
        powerlines_cfg.name = f"{powerlines_cfg.name}-fold-{fold}"
        powerlines_cfg.data.cv.fold = fold

    if args.validate_run is None:
        if args.resume is not None:
            assert args.resume_epoch is not None, "--resume_epoch must be specified when resuming the run"
            powerlines_cfg.checkpoint.resume = True
            powerlines_cfg.checkpoint.run_id = args.resume
            powerlines_cfg.checkpoint.epoch = args.resume_epoch

        run_training(powerlines_cfg)
    else:
        run_validation(args.validate_run, powerlines_cfg)
