'''
by lyuwenyu
'''
from typing import Any, Dict, Optional

import neptune

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
            formatted_name = name[8:]  # without "metrics/" part
            for entity, metric in value.items():
                run[f"metrics/{subset}/{formatted_name}/{entity}"].log(metric)
        else:
            run[f"metrics/{subset}/misc/{name}"].log(value)


class DetSolver(BaseSolver):
    def fit(self) -> Optional[float]:
        print("Start training")
        self.train()

        args = self.cfg
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_epochs = self.cfg_powerlines.epochs
        print('number of params:', n_parameters)

        for epoch in range(self.last_epoch + 1, n_epochs):
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

            if self.cfg_powerlines.lr_scheduler.enabled:
                self.lr_scheduler.step()

            # Log train metrics and checkpoint
            log_stats(self.run, "train", train_stats)
            self.save_checkpoint(epoch)

            # Validate
            model = self.ema.module if self.ema else self.model
            if self.cfg_powerlines.validation.every:
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
                log_stats(self.run, "val", val_stats)

        if self.cfg_powerlines.validation.last:
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
            log_stats(self.run, "val", val_stats)
            return val_stats[self.cfg_powerlines.optimized_metric]
        else:
            return None

    def val(self):
        self.eval()

        model = self.ema.module if self.ema else self.model
        val_stats = evaluate(
            0, self.cfg_powerlines, model, self.criterion, self.postprocessor, self.val_dataloader, self.device, self.run
        )
        log_stats(self.run, "val", val_stats)
