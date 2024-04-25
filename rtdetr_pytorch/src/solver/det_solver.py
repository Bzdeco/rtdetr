'''
by lyuwenyu
'''
from typing import Any, Dict, Optional

import neptune
import numpy as np

from src.misc import dist

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


def format_stats_as_flat_metrics_dict(subset: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {}
    for name, value in stats.items():
        if name.startswith("loss"):
            if name == "loss":
                metrics[f"metrics/{subset}/loss/total"] = value
            else:
                loss_component_name = name[name.find("_") + 1:]
                metrics[f"metrics/{subset}/loss/{loss_component_name}"] = value
        elif name.startswith("metrics"):
            formatted_name = name[8:]  # without "metrics/" part
            for entity, metric_value in value.items():
                metrics[f"metrics/{subset}/{formatted_name}/{entity}"] = metric_value
        else:
            metrics[f"metrics/{subset}/misc/{name}"] = value

    return metrics


def log_metrics(run: neptune.Run, metrics: Dict[str, Any]):
    for metric, value in metrics.items():
        run[metric].log(value)


class DetSolver(BaseSolver):
    def fit(self) -> Optional[float]:
        print("Start training")
        self.train()

        args = self.cfg
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_epochs = self.cfg_powerlines.epochs
        print('number of params:', n_parameters)

        best_metric_value = -np.inf
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
            train_metrics = format_stats_as_flat_metrics_dict("train", train_stats)
            log_metrics(self.run, train_metrics)
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
                val_metrics = format_stats_as_flat_metrics_dict("val", val_stats)
                log_metrics(self.run, val_metrics)
                best_metric_value = max(best_metric_value, val_metrics[self.cfg_powerlines.optimized_metric])

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
            val_metrics = format_stats_as_flat_metrics_dict("val", val_stats)
            log_metrics(self.run, val_metrics)
            return val_metrics[self.cfg_powerlines.optimized_metric]
        else:
            return best_metric_value

    def val(self):
        self.eval()

        model = self.ema.module if self.ema else self.model
        val_stats = evaluate(
            0, self.cfg_powerlines, model, self.criterion, self.postprocessor, self.val_dataloader, self.device, self.run
        )
        val_metrics = format_stats_as_flat_metrics_dict("val", val_stats)
        log_metrics(self.run, val_metrics)
