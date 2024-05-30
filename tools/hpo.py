from utils import seed
seed.set_global_seeds()

# import torch.multiprocessing
# torch.multiprocessing.set_start_method("spawn", force=True)

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


def _values_range(base: float, spread: float) -> List[float]:
    return [base - spread, base + spread]


PATCH_SIZE = 1024


def perturbation_from_hyperparameters(hyperparameters: Union[Configuration, Dict[str, Any]]) -> int:
    return int(hyperparameters["perturbation_fraction"] * PATCH_SIZE)


def overrides_from_trial_config(hpo_run_id: int, trial_id: int) -> List[str]:
    hpo_run = fetch_run("jakubg/powerlines", hpo_run_id)
    hyperparameters = hpo_run[f"trials/{trial_id}"].fetch()
    return [
        f"data.augmentations.color_jitter.magnitude={hyperparameters['color_jitter_magnitude']}",
        f"data.augmentations.multi_scale.enabled={hyperparameters['multi_scale_enabled']}",
        f"data.perturbation={perturbation_from_hyperparameters(hyperparameters)}",
        f"data.negative_sample_prob={hyperparameters['negative_sample_prob']}",
        f"data.batch_size.train={hyperparameters['batch_size']}",
        f"loss.ohem.enabled={hyperparameters['ohem_enabled']}",
        f"optimizer.lr={hyperparameters['lr']}",
        f"optimizer.wd={hyperparameters['wd']}"
    ]


def overrides_from_hpc(
    config: Configuration, epochs: int
) -> List[str]:
    return [
        f"data.augmentations.color_jitter.magnitude={config['color_jitter_magnitude']}",
        f"data.augmentations.multi_scale.enabled={config['multi_scale_enabled']}",
        f"data.perturbation={perturbation_from_hyperparameters(config)}",
        f"data.negative_sample_prob={config['negative_sample_prob']}",
        f"data.batch_size.train={config['batch_size']}",
        f"loss.ohem.enabled={config['ohem_enabled']}",
        f"optimizer.lr={config['lr']}",
        f"optimizer.wd={config['wd']}",
        f"epochs={epochs}"
    ]


class HPORunner:
    def __init__(
        self,
        name: str,
        goal: str = "maximize"
    ):
        self.name = name
        self._goal = goal

    def configuration_space(self) -> ConfigurationSpace:
        config_space = ConfigurationSpace()

        config_space.add_hyperparameters([
            Float("color_jitter_magnitude", (0.0, 0.5), default=0.2),
            Categorical("multi_scale_enabled", [False, True], default=True),
            Float("perturbation_fraction", (0.0, 0.875), default=0.375),
            Float("negative_sample_prob", (0.0, 0.25), default=0.12),
            Integer("batch_size", (2, 64), default=12, log=True),
            Categorical("ohem_enabled", [False, True], default=True),
            Float("lr", (1e-5, 1e-2), default=1e-2, log=True),
            Float("wd", (1e-6, 1e-1), default=5e-4, log=True),
        ])

        return config_space

    def hydra_config_from_hpc(self, config: Configuration, epochs: int) -> DictConfig:
        with initialize(version_base=None, config_path="../configs/powerlines", job_name=self.name):
            overrides = overrides_from_hpc(config, epochs)
            print(f"Trial overrides: {overrides}")
            hydra_config = compose(config_name="config", overrides=overrides)
            hydra_config.name = f"{self.name}"
            return hydra_config

    def default_config(self) -> DictConfig:
        with initialize(version_base=None, config_path="../configs/powerlines", job_name=self.name):
            config = compose(config_name="config")
            return config

    def target_function(self, config: Configuration, seed: int = 0, budget: int = 5) -> Dict[str, float]:
        # Train model and get best achieved result
        torch.cuda.empty_cache()
        hydra_config = self.hydra_config_from_hpc(config, epochs=int(budget))

        optimized_metrics = run_training(hydra_config)

        if self._goal == "maximize":
            return {
                name: 1 - value  # SMAC minimizes the target function
                for name, value in optimized_metrics.items()
            }
        else:
            return optimized_metrics


def run_hyper_parameter_search(
    name: str,
    optimized_metrics: List[str],
    output_directory: Path,
    n_trials: int,
    n_workers: int,
    min_epochs: int,
    max_epochs: int,
    n_initial_designs: int,
    resume: bool = False,
):
    torch.cuda.empty_cache()

    hpo_runner = HPORunner(name)

    scenario = Scenario(
        hpo_runner.configuration_space(),
        name=hpo_runner.name,
        objectives=optimized_metrics,
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
    parser.add_argument("--metrics", default=["cables/quality", "poles/quality"])
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
