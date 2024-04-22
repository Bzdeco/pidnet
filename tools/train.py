# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os

from pathlib import Path
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from hydra import initialize, compose
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import models
from configs import config
from configs import update_config
from tools import factory
from utils.criterion import CrossEntropy, OhemCrossEntropy, BoundaryLoss
from utils.function import train, validate
from utils.utils import FullModel, checkpoint_folder, create_neptune_run, run_id


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def powerlines_config() -> DictConfig:
    with initialize(version_base=None, config_path="../configs/powerlines"):
        return compose(config_name="config")


def main():
    args = parse_args()
    config_powerlines = powerlines_config()

    if args.seed > 0:
        import random
        print("Seeding with", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    run = create_neptune_run(config_powerlines.name, resume=False, from_run_id=None)
    output_folder = checkpoint_folder(config_powerlines, run_id(run))
    run["config/powerlines"] = stringify_unsupported(config_powerlines)
    run["config/pidnet"] = stringify_unsupported(dict(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    train_dataset = factory.train_dataset(config_powerlines)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True
    )

    val_dataloader = DataLoader(
        factory.val_dataset(config_powerlines),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False
    )

    # criterion
    if config.LOSS.USE_OHEM:
        semantic_seg_criterion = OhemCrossEntropy(
            ignore_label=config.TRAIN.IGNORE_LABEL,
            thres=config.LOSS.OHEMTHRES,
            min_kept=config.LOSS.OHEMKEEP,
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
    optimizer = factory.optimizer(config, model)

    epoch_iters = int(len(train_dataset) / config.TRAIN.BATCH_SIZE_PER_GPU)
        
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

    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        train(run, config_powerlines, epoch, epoch_iters, config.TRAIN.LR, num_iters, train_dataloader, optimizer, model)
        validate(config, run, val_dataloader, model)
        save_checkpoint(epoch, output_folder, model, optimizer)


def save_checkpoint(epoch: int, folder: Path, model: nn.Module, optimizer: Optimizer, filename: Optional[str] = None):
    filename = filename or f"{epoch:03d}.pt"
    torch.save({
        "epoch": epoch + 1,
        "state_dict": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, folder / filename)


if __name__ == '__main__':
    main()
