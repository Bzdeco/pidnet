from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.dataset.inference import InferenceCablesDetectionDataset
from powerlines.data.dataset.sampling import TrainCablesDetectionDataset


def data_source(config: DictConfig, subset: str) -> DataSourceConfig:
    paths_config = config.paths
    return DataSourceConfig(
        complete_frames_root_folder=Path(paths_config.complete_frames),
        annotations_path=Path(paths_config.annotations),
        data_source_subset=subset,
        cv_config=config.data.cv
    )


def loading(config: DictConfig) -> LoadingConfig:
    return LoadingConfig(
        distance_mask=True,
        exclusion_area_mask=True,
        labels=True
    )


def sampling(config: DictConfig) -> SamplingConfig:
    data_config = config.data
    return SamplingConfig(
        patch_size=data_config.patch_size,
        perturbation_size=data_config.perturbation,
        negative_sample_prob=data_config.negative_sample_prob,
        non_sky_bias=data_config.non_sky_bias,
        remove_excluded_area=True
    )


def train_dataset(config: DictConfig) -> Dataset:
    return TrainCablesDetectionDataset(
        data_config=config.data,
        data_source=data_source(config, "train"),
        loading=loading(config),
        sampling=sampling(config),
        num_frames=config.data.size.train
    )


def val_dataset(config: DictConfig) -> Dataset:
    return InferenceCablesDetectionDataset(
        data_config=config.data,
        data_source=data_source(config, "val"),
        loading=loading(config),
        sampling=sampling(config),
        num_frames=config.data.size.val
    )


def optimizer(config_powerlines: DictConfig, model: nn.Module):
    params_dict = dict(model.named_parameters())
    optimizer_config = config_powerlines.optimizer

    params = [{"params": list(params_dict.values()), "lr": optimizer_config.lr}]

    if optimizer_config.name == "sgd":
        return torch.optim.SGD(
            params,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.wd,
            nesterov=optimizer_config.nesterov,
        )
    elif optimizer_config.name == "adam":
        return torch.optim.Adam(
            params,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.wd
        )
    else:
        raise ValueError(f"Unsupported optimized {optimizer_config.name}")
