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
        exclusion_area_mask=True
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
        data_source=data_source(config, "train"),
        loading=loading(config),
        sampling=sampling(config),
        num_frames=config.data.size.train
    )


def val_dataset(config: DictConfig) -> Dataset:
    return InferenceCablesDetectionDataset(
        data_source=data_source(config, "val"),
        loading=loading(config),
        num_frames=config.data.size.val
    )


def optimizer(config, model: nn.Module):
    params_dict = dict(model.named_parameters())
    params = [{"params": list(params_dict.values()), "lr": config.TRAIN.LR}]

    if config.TRAIN.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        return torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WD
        )
    else:
        raise ValueError(f"Unsupported optimized {config.TRAIN.OPTIMIZER}")
