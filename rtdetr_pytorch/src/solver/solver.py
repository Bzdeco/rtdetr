"""by lyuwenyu
"""

import torch 

from datetime import datetime
from pathlib import Path 
from typing import Dict

from neptune.utils import stringify_unsupported
from omegaconf import DictConfig

from powerlines import factory
from powerlines.logger import create_neptune_run, run_id
from src.misc import dist
from src.core import BaseConfig


CHECKPOINTS_FOLDER = Path("/scratch/cvlab/home/gwizdala/output/checkpoints/")


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig, cfg_powerlines: DictConfig) -> None:
        self.cfg = cfg
        self.cfg_powerlines = cfg_powerlines

        # Set resumption checkpoint path
        checkpoint_config = cfg_powerlines.checkpoint
        if checkpoint_config.resume:
            assert checkpoint_config.epoch is not None, "Checkpoint epoch not set"
            filename = self._filename(checkpoint_config.epoch)
            self.cfg.resume = str(CHECKPOINTS_FOLDER / str(cfg_powerlines.checkpoint.run_id) / filename)
            print(f"Resuming experiment {cfg_powerlines.checkpoint.run_id} from {self.cfg.resume}")

        self.run = create_neptune_run(
            name=cfg_powerlines.name,
            resume=cfg_powerlines.checkpoint.resume,
            from_run_id=cfg_powerlines.checkpoint.run_id
        )
        self.run["config"] = stringify_unsupported(self.cfg_powerlines)

        self.output_dir = CHECKPOINTS_FOLDER / str(run_id(self.run))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        '''Avoid instantiating unnecessary classes 
        '''
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor

        # NOTE (lvwenyu): should load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None

    def train(self):
        self.setup()
        self.optimizer = factory.optimizer(self.cfg_powerlines, self.model)
        self.lr_scheduler = factory.lr_scheduler(self.cfg_powerlines, self.optimizer)

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        self.train_dataloader = factory.dataloader(
            factory.train_dataset(self.cfg_powerlines),
            batch_size=self.cfg_powerlines.data.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg_powerlines.data.num_workers.train
        )
        self.val_dataloader = factory.dataloader(
            factory.val_dataset(self.cfg_powerlines),
            batch_size=1,  # fix to 1 full resolution image which gets sliced
            drop_last=False,
            shuffle=True,
            num_workers=self.cfg_powerlines.data.num_workers.val
        )

    def eval(self):
        self.setup()
        self.val_dataloader = factory.dataloader(
            factory.val_dataset(self.cfg_powerlines),
            batch_size=1,  # fix to 1 full resolution image which gets sliced
            drop_last=False,
            shuffle=True,
            num_workers=self.cfg_powerlines.data.num_workers.val
        )

        if self.cfg.resume:
            print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)

    def _filename(self, epoch: int) -> str:
        return f"{epoch:03d}.pt"

    def save_checkpoint(self, epoch: int):
        if self.output_dir:
            checkpoint_paths = [self.output_dir / self._filename(epoch)]
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)

    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            # state['last_epoch'] = self.lr_scheduler.last_epoch

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state


    def load_state_dict(self, state):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')

    def save(self, path):
        '''save state
        '''
        state = self.state_dict()
        dist.save_on_master(state, path)

    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def load_tuning_state(self, path,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if 'http' in path:
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def fit(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError
