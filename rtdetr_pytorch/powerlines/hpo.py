import os

import neptune
from neptune import Run
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from smac import Callback
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue


def fetch_run(project: str, run_id: int, mode: str = "read-only"):
    assert mode in {"read-only", "sync", "async", "debug"}

    return neptune.init_run(
        project=project,
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        with_id=f"POW-{run_id}",
        mode=mode
    )


def create_neptune_hpo_run(config: DictConfig, debug: bool = False) -> Run:
    return neptune.init_run(
        project="jakubg/powerlines",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        name=config.name,
        tags=["hpo"],
        mode="async" if not debug else "debug",
        capture_stdout=True,
        capture_stderr=True,
        capture_traceback=False,
        capture_hardware_metrics=True,
        flush_period=10
    )


class HyperparameterOptimizationCallback(Callback):
    """
    Based on neptune integration with Optuna: https://neptune.ai/resources/optuna-integration
    """

    def __init__(self, hpo_name: str, config: DictConfig, debug: bool = False):
        super().__init__()
        hpo_config = config.copy()
        hpo_config.name = hpo_name
        self._run = create_neptune_hpo_run(hpo_config, debug)

        self._current_trial = -1
        self._current_config_dict = None

    def on_ask_end(self, smbo: SMBO, info: TrialInfo) -> None:
        pass

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue):
        self._current_trial += 1

        # Log trial configuration
        trial_result = 1 - value.cost
        trial_config = dict(info.config)
        epochs = int(info.budget)
        self._run[f"trials/{self._current_trial}"] = stringify_unsupported(trial_config)
        self._run[f"trials/{self._current_trial}/epochs"] = epochs

        # Log trial results
        self._run[f"trials/{self._current_trial}/result"] = trial_result
        self._run[f"results"].log(trial_result)
        self._run.sync()

        # Update incumbent trial
        incumbent = smbo.intensifier.get_incumbent()
        if incumbent is not None and dict(incumbent) == trial_config:
            self._run["incumbent_trial"] = self._current_trial
