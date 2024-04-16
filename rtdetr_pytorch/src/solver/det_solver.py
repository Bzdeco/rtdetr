'''
by lyuwenyu
'''
from typing import Any, Dict

import neptune
from neptune.utils import stringify_unsupported

from src.misc import dist

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


def log_stats(run: neptune.Run, subset: str, stats: Dict[str, Any]):
    for name, value in stats.items():
        if name.startswith("loss"):
            if name == "loss":
                run[f"metrics/{subset}/loss/total"].log(value)
            else:
                loss_component_name = name[name.find("_") + 1:]
                run[f"metrics/{subset}/loss/{loss_component_name}"].log(value)
        elif name.startswith("metrics"):
            run[f"metrics/{subset}/{name[8:]}"].log(stringify_unsupported(value))
        else:
            run[f"metrics/{subset}/misc/{name}"].log(value)


class DetSolver(BaseSolver):
    def fit(self):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        for epoch in range(self.last_epoch + 1, args.epochs):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # Train single epoch
            train_stats = train_one_epoch(
                self.cfg_powerlines,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                max_norm=args.clip_max_norm,
                ema=self.ema,
                scaler=self.scaler)
            self.lr_scheduler.step()

            # Train metrics and checkpoint
            log_stats(self.run, "train", train_stats)
            self.save_checkpoint(epoch)

            # Validate
            model = self.ema.module if self.ema else self.model
            val_stats = evaluate(
                epoch,
                self.cfg_powerlines,
                model,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.device,
                self.run
            )

            # Validation metrics
            log_stats(self.run, "val", val_stats)

    def val(self):
        self.eval()

        model = self.ema.module if self.ema else self.model
        val_stats = evaluate(
            model, self.criterion, self.postprocessor, self.val_dataloader, self.device
        )
        log_stats(self.run, "val", val_stats)

    def save_checkpoint(self, epoch: int):
        if self.output_dir:
            checkpoint_paths = [self.output_dir / f'{epoch:03d}.pt']
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(self.state_dict(epoch), checkpoint_path)
