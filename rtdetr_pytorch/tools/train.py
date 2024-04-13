"""by lyuwenyu and Bzdeco
"""
from powerlines.data.seed import set_global_seeds

# TODO: set torch generator and seed worker seeds
set_global_seeds()

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args) -> None:
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="configs/rtdetr/rtdetr_r18vd_6x_coco.yml")
    parser.add_argument('--tuning', '-t', type=str, default="checkpoints/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth")
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=True)

    args = parser.parse_args()

    main(args)
