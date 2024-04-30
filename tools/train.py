# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import argparse
import os

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from hydra import initialize, compose
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from torch import nn
from torch.cuda import amp
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import models
from configs import config
from configs import update_config
from tools import factory
from utils.criterion import CrossEntropy, OhemCrossEntropy, BoundaryLoss
from utils.function import train, validate
from utils.utils import FullModel, checkpoint_folder, create_neptune_run, run_id


SEED = 304


def powerlines_config() -> DictConfig:
    with initialize(version_base=None, config_path="../configs/powerlines"):
        return compose(config_name="config")


def default_commandline_arguments() -> DictConfig:
    return DictConfig({
        "cfg": "configs/powerlines/pidnet_small_powerlines.yaml",
        "seed": SEED,
        "opts": []
    })


def run_training(config_powerlines: DictConfig) -> Optional[float]:  # returns optimized metric
    args = default_commandline_arguments()
    update_config(config, args)

    import random
    print("Seeding with", SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    run = create_neptune_run(config_powerlines.name, resume=False, from_run_id=None)
    output_folder = checkpoint_folder(config_powerlines, run_id(run))
    run["config"] = stringify_unsupported(config_powerlines)
    run["config/pidnet"] = stringify_unsupported(dict(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    train_dataset = factory.train_dataset(config_powerlines)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config_powerlines.data.batch_size.train,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config_powerlines.data.num_workers.train,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        factory.val_dataset(config_powerlines),
        batch_size=config_powerlines.data.batch_size.val,
        shuffle=False,
        num_workers=config_powerlines.data.num_workers.val,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False
    )

    # criterion
    ohem_config = config_powerlines.loss.ohem
    if ohem_config.enabled:
        print("Using OHEM in loss function")
        min_kept = int(ohem_config.keep_fraction * (config_powerlines.data.patch_size ** 2))
        semantic_seg_criterion = OhemCrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            thres=ohem_config.threshold,
            min_kept=min_kept,
            weight=train_dataset.class_weights
        )
    else:
        semantic_seg_criterion = CrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            weight=train_dataset.class_weights
        )
    bd_criterion = BoundaryLoss()

    pidnet = models.pidnet.get_seg_model(config)  # creates a pretrained model by default
    model = FullModel(pidnet, semantic_seg_criterion, bd_criterion).cuda()
    optimizer = factory.optimizer(config_powerlines, model)
    scaler = amp.GradScaler(enabled=True)

    epoch_iters = int(len(train_dataset) / config_powerlines.data.batch_size.train)

    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(output_folder, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    n_epochs = config_powerlines.epochs
    num_iters = n_epochs * epoch_iters
    validation_config = config_powerlines.validation

    best_metric_value = -np.inf
    for epoch in range(last_epoch, n_epochs):
        train(run, config_powerlines, epoch, epoch_iters, num_iters, train_dataloader, optimizer, scaler, model)
        if validation_config.every:
            metric_value = validate(n_epochs - 1, config, config_powerlines, run, val_dataloader, model)
            best_metric_value = max(best_metric_value, metric_value)
        save_checkpoint(epoch, output_folder, model, optimizer)

    if validation_config.last:
        return validate(n_epochs - 1, config, config_powerlines, run, val_dataloader, model)
    else:
        return best_metric_value


def save_checkpoint(epoch: int, folder: Path, model: nn.Module, optimizer: Optimizer, filename: Optional[str] = None):
    filename = filename or f"{epoch:03d}.pt"
    torch.save({
        "epoch": epoch + 1,
        "state_dict": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, folder / filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=False, type=int)
    args = parser.parse_args()

    # Set k-fold CV fold if given
    powerlines_cfg = powerlines_config()
    if args.fold is not None:
        fold = int(args.fold)
        powerlines_cfg.cv_name = powerlines_cfg.name
        powerlines_cfg.name = f"{powerlines_cfg.name}-fold-{fold}"
        powerlines_cfg.data.cv.fold = fold

    run_training(powerlines_cfg)
