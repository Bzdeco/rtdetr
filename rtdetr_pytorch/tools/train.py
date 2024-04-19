"""by lyuwenyu and Bzdeco
"""
from powerlines.data.seed import set_global_seeds
set_global_seeds()

from hydra import initialize, compose
from omegaconf import DictConfig
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import DetSolver


def rt_detr_config(args: argparse.Namespace) -> YAMLConfig:
    return YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )


def powerlines_config() -> DictConfig:
    with initialize(version_base=None, config_path="../configs"):
        return compose(config_name="powerlines")


def main(args) -> None:
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    # Create and configure solver
    cfg = rt_detr_config(args)
    cfg_powerlines = powerlines_config()
    solver = DetSolver(cfg, cfg_powerlines)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="configs/rtdetr/rtdetr_r18vd_6x_coco.yml")
    parser.add_argument('--tuning', '-t', type=str, default="checkpoints/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth")
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=True)

    main(parser.parse_args())
