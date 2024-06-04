# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import argparse

from pathlib import Path
from typing import Optional, Dict

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from hydra import initialize, compose
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from torch import nn, ModuleDict
from torch.cuda import amp
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import models
from configs import config
from configs import update_config
from powerlines.data import seed
from tools import factory
from utils.criterion import CrossEntropy, OhemCrossEntropy
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


# returns optimized metric
def run_training(
    config_powerlines: DictConfig,
    resume_run_id: Optional[int] = None,
    resume_epoch: Optional[int] = None
) -> Optional[Dict[str, float]]:
    update_config(config, default_commandline_arguments())

    import random
    print("Seeding with", SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    resume = (resume_run_id is not None)
    run = create_neptune_run(config_powerlines.name, resume=resume, from_run_id=resume_run_id)
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
        drop_last=True,
        worker_init_fn=seed.seed_worker,
        generator=seed.torch_generator()
    )

    val_dataloader = DataLoader(
        factory.val_dataset(config_powerlines),
        batch_size=config_powerlines.data.batch_size.val,
        shuffle=False,
        num_workers=config_powerlines.data.num_workers.val,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
        worker_init_fn=seed.seed_worker,
        generator=seed.torch_generator()
    )

    # criterion
    ohem_config = config_powerlines.loss.ohem
    if ohem_config.enabled:
        print("Using OHEM in loss function")
        min_kept = int(ohem_config.keep_fraction * (config_powerlines.data.patch_size ** 2))
        semantic_seg_criterion = {
            "cables": OhemCrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                thres=ohem_config.threshold,
                min_kept=min_kept,
                weight=train_dataset.cables_class_weights
            ),
            "poles": OhemCrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                thres=ohem_config.threshold,
                min_kept=min_kept,
                weight=train_dataset.poles_class_weights
            )
        }
    else:
        semantic_seg_criterion = {
            "cables": CrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                weight=train_dataset.cables_class_weights
            ),
            "poles": CrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                weight=train_dataset.poles_class_weights
            )
        }

    pidnet = models.pidnet.get_seg_model(config)  # creates a pretrained model by default
    model = FullModel(pidnet, semantic_seg_criterion, poles_weight=config_powerlines.loss.poles_weight).cuda()
    optimizer = factory.optimizer(config_powerlines, model)
    scaler = amp.GradScaler(enabled=True)

    first_epoch = 0
    if resume:
        model, optimizer, scaler = load_checkpoint(resume_epoch, output_folder, model, optimizer, scaler)
        first_epoch = resume_epoch + 1

    epoch_iters = int(len(train_dataset) / config_powerlines.data.batch_size.train)
    n_epochs = config_powerlines.epochs
    num_iters = n_epochs * epoch_iters
    validation_config = config_powerlines.validation

    best_metric_values = None
    for epoch in range(first_epoch, n_epochs):
        train(run, config_powerlines, epoch, epoch_iters, num_iters, train_dataloader, optimizer, scaler, model)
        if validation_config.every:
            optimized_metrics = validate(epoch, config, config_powerlines, run, val_dataloader, model)
            if best_metric_values is None:
                best_metric_values = optimized_metrics
            else:
                best_metric_values = {
                    metric_name: max(best_metric_values[metric_name], optimized_metrics[metric_name])
                    for metric_name in optimized_metrics.keys()
                }
        save_checkpoint(epoch, output_folder, model, optimizer, scaler)

    if validation_config.last:
        return validate(n_epochs - 1, config, config_powerlines, run, val_dataloader, model)
    else:
        return best_metric_values


def run_validation(
    config_powerlines: DictConfig,
    resume_run_id: Optional[int] = None
):
    update_config(config, default_commandline_arguments())

    import random
    print("Seeding with", SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    resume = (resume_run_id is not None)
    run = create_neptune_run(config_powerlines.name, resume=resume, from_run_id=resume_run_id)
    output_folder = checkpoint_folder(config_powerlines, run_id(run))
    run["config"] = stringify_unsupported(config_powerlines)
    run["config/pidnet"] = stringify_unsupported(dict(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    train_dataset = factory.train_dataset(config_powerlines)

    val_dataloader = DataLoader(
        factory.val_dataset(config_powerlines),
        batch_size=config_powerlines.data.batch_size.val,
        shuffle=False,
        num_workers=config_powerlines.data.num_workers.val,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
        worker_init_fn=seed.seed_worker,
        generator=seed.torch_generator()
    )

    # criterion
    ohem_config = config_powerlines.loss.ohem
    if ohem_config.enabled:
        print("Using OHEM in loss function")
        min_kept = int(ohem_config.keep_fraction * (config_powerlines.data.patch_size ** 2))
        semantic_seg_criterion = {
            "cables": OhemCrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                thres=ohem_config.threshold,
                min_kept=min_kept,
                weight=train_dataset.cables_class_weights
            ),
            "poles": OhemCrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                thres=ohem_config.threshold,
                min_kept=min_kept,
                weight=train_dataset.poles_class_weights
            )
        }
    else:
        semantic_seg_criterion = {
            "cables": CrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                weight=train_dataset.cables_class_weights
            ),
            "poles": CrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                weight=train_dataset.poles_class_weights
            )
        }

    pidnet = models.pidnet.get_seg_model(config)  # creates a pretrained model by default
    model = FullModel(pidnet, semantic_seg_criterion, poles_weight=config_powerlines.loss.poles_weight).cuda()
    optimizer = factory.optimizer(config_powerlines, model)
    scaler = amp.GradScaler(enabled=True)

    checkpoints = sorted(list(output_folder.glob("*.pt")))
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem)
        model, optimizer, scaler = load_checkpoint(epoch, output_folder, model, optimizer, scaler)
        validate(epoch, config, config_powerlines, run, val_dataloader, model)


def save_checkpoint(
    epoch: int, folder: Path, model: nn.Module, optimizer: Optimizer, scaler: GradScaler, filename: Optional[str] = None
):
    filename = filename or f"{epoch:03d}.pt"
    torch.save({
        "epoch": epoch + 1,
        "state_dict": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict()
    }, folder / filename)


def load_checkpoint(epoch: int, folder: Path, model: nn.Module, optimizer: Optimizer, scaler: GradScaler):
    filepath = folder / f"{epoch:03d}.pt"
    print(f"Loading checkpoint from {filepath}")

    checkpoint = torch.load(filepath)
    model.model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return model, optimizer, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=False, type=int)
    parser.add_argument("--resume", required=False, type=int)
    parser.add_argument("--validate", required=False, type=int)
    parser.add_argument("--epoch", required=False, type=int)
    args = parser.parse_args()

    # Set k-fold CV fold if given
    powerlines_cfg = powerlines_config()
    if args.fold is not None:
        fold = int(args.fold)
        powerlines_cfg.cv_name = powerlines_cfg.name
        powerlines_cfg.name = f"{powerlines_cfg.name}-fold-{fold}"
        powerlines_cfg.data.cv.fold = fold

    if args.validate is not None:
        assert args.fold is not None, "Must provide fold of the validated run"
        run_validation(powerlines_cfg, args.validate)
    else:
        run_training(powerlines_cfg, args.resume, args.epoch)