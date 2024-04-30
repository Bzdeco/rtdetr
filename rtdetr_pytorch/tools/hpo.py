from powerlines.data import seed
seed.set_global_seeds()

from powerlines.hpo import HyperparameterOptimizationCallback, fetch_run
from tools.train import run_training

import argparse

from pathlib import Path
from typing import List, Dict, Any, Union

import torch
from omegaconf import DictConfig
from ConfigSpace import ConfigurationSpace, Configuration, Float, Integer, Categorical
from hydra import initialize, compose
from smac import Scenario, MultiFidelityFacade
from smac.intensifier import Hyperband


PATCH_SIZE = 1024


def _values_range(base: float, spread: float) -> List[float]:
    return [base - spread, base + spread]


def perturbation_from_hyperparameters(hyperparameters: Union[Configuration, Dict[str, Any]]) -> int:
    return int(hyperparameters["perturbation_fraction"] * PATCH_SIZE)


def overrides_from_trial_config(hpo_run_id: int, trial_id: int) -> List[str]:
    hpo_run = fetch_run("jakubg/powerlines", hpo_run_id)
    hyperparameters = hpo_run[f"trials/{trial_id}"].fetch()
    return [
        f"data.augmentations.cj_magnitude={hyperparameters['cj_magnitude']}",
        f"data.augmentations.multi_scale_prob={hyperparameters['multi_scale_prob']}",
        f"data.perturbation={perturbation_from_hyperparameters(hyperparameters)}",
        f"data.negative_sample_prob={hyperparameters['negative_sample_prob']}",
        f"data.batch_size={hyperparameters['batch_size']}",
        f"optimizer.lr={hyperparameters['lr']}",
        f"optimizer.lr_backbone={hyperparameters['lr_backbone']}",
        f"optimizer.wd={hyperparameters['wd']}",
        f"lr_scheduler.enabled={hyperparameters['lr_scheduler_enabled']}"
    ]


def overrides_from_hpc(
    config: Configuration, epochs: int
) -> List[str]:
    return [
        f"data.augmentations.cj_magnitude={config['cj_magnitude']}",
        f"data.augmentations.multi_scale_prob={config['multi_scale_prob']}",
        f"data.perturbation={perturbation_from_hyperparameters(config)}",
        f"data.negative_sample_prob={config['negative_sample_prob']}",
        f"data.batch_size={config['batch_size']}",
        f"optimizer.lr={config['lr']}",
        f"optimizer.lr_backbone={config['lr_backbone']}",
        f"optimizer.wd={config['wd']}",
        f"lr_scheduler.enabled={config['lr_scheduler_enabled']}",
        f"epochs={epochs}"
    ]


class HPORunner:
    def __init__(
        self,
        name: str,
        optimized_metric: str,
        goal: str = "maximize"
    ):
        self.name = name
        self._optimized_metric = optimized_metric
        self._goal = goal

    def configuration_space(self) -> ConfigurationSpace:
        config_space = ConfigurationSpace()

        config_space.add_hyperparameters([
            Float("cj_magnitude", (0.0, 0.3), default=0.2),
            Float("multi_scale_prob", (0.0, 0.8), default=0.5),
            Float("perturbation_fraction", (0.0, 0.875), default=0.375),
            Float("negative_sample_prob", (0.0, 0.35), default=0.12),
            Integer("batch_size", (2, 16), default=16, log=True),
            Float("lr", (1e-5, 1e-2), default=1e-4, log=True),
            Float("lr_backbone", (1e-6, 1e-3), default=1e-5, log=True),
            Float("wd", (1e-6, 1e-1), default=1e-4, log=True),
            Categorical("lr_scheduler_enabled", [False, True], default=True)
        ])

        return config_space

    def hydra_config_from_hpc(self, config: Configuration, epochs: int) -> DictConfig:
        with initialize(version_base=None, config_path="../configs", job_name=self.name):
            overrides = overrides_from_hpc(config, epochs)
            print(f"Trial overrides: {overrides}")
            hydra_config = compose(config_name="powerlines", overrides=overrides)
            hydra_config.name = f"{self.name}"
            hydra_config.optimized_metric = self._optimized_metric
            return hydra_config

    def default_config(self) -> DictConfig:
        with initialize(version_base=None, config_path="../configs", job_name=self.name):
            config = compose(config_name="powerlines")
            config.optimized_metric = self._optimized_metric
            return config

    def target_function(self, config: Configuration, seed: int = 0, budget: int = 5) -> float:
        # Train model and get best achieved result
        torch.cuda.empty_cache()
        hydra_config = self.hydra_config_from_hpc(config, epochs=int(budget))

        optimized_metric_value = run_training(hydra_config)

        if self._goal == "maximize":
            return 1 - optimized_metric_value  # SMAC minimizes the target function
        else:
            return optimized_metric_value


def run_hyper_parameter_search(
    name: str,
    optimized_metric: str,
    output_directory: Path,
    n_trials: int,
    n_workers: int,
    min_epochs: int,
    max_epochs: int,
    n_initial_designs: int,
    resume: bool = False,
):
    torch.cuda.empty_cache()

    hpo_runner = HPORunner(name, optimized_metric)

    scenario = Scenario(
        hpo_runner.configuration_space(),
        name=hpo_runner.name,
        output_directory=output_directory,
        deterministic=True,
        n_trials=n_trials,
        use_default_config=True,
        min_budget=min_epochs,
        max_budget=max_epochs,
        n_workers=n_workers
    )
    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=n_initial_designs)
    intensifier = Hyperband(scenario, eta=2, incumbent_selection="highest_budget")

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=hpo_runner.target_function,
        initial_design=initial_design,
        intensifier=intensifier,
        callbacks=[HyperparameterOptimizationCallback(hpo_runner.name, hpo_runner.default_config())],
        overwrite=(not resume)
    )
    incumbent = smac.optimize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="metrics/val/ccq/masked/quality/0.004")
    parser.add_argument("--n_trials", default=500)
    parser.add_argument("--n_workers", default=1)
    parser.add_argument("--min_epochs", default=2)
    parser.add_argument("--max_epochs", default=10)
    parser.add_argument("--n_initial_designs", default=5)
    parser.add_argument("--resume", default=False, action="store_true")
    args = parser.parse_args()

    name = args.name
    metric = args.metric
    output = Path(args.output)
    n_trials = int(args.n_trials)
    n_workers = int(args.n_workers)
    n_initial_designs = int(args.n_initial_designs)
    min_epochs, max_epochs = int(args.min_epochs), int(args.max_epochs)
    resume = bool(args.resume)
    print(
        f"Hyperparameter optimization - {name}:\n",
        f"metric={metric}\n"
        f"output={output}\n"
        f"trials={n_trials}\n"
        f"workers={n_workers}\n"
        f"initial_designs={n_initial_designs}\n"
        f"epochs=({min_epochs}, {max_epochs})\n"
        f"resume={resume}\n"
    )

    run_hyper_parameter_search(
        name,
        metric,
        output,
        n_trials,
        n_workers,
        min_epochs,
        args.max_epochs,
        n_initial_designs,
        resume=resume,
    )
