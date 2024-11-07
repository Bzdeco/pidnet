import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tensorflow.python.ops import nn
from torch.utils.data import DataLoader

import models
from configs import config
from powerlines.data import seed
from tools import factory
from tools.train import powerlines_config
from utils.criterion import OhemCrossEntropy, CrossEntropy
from utils.function import validate
from utils.utils import FullModel


def cross_validation_metrics(metrics: Dict[str, List[float]]):
    aggregated_metrics = {}
    for name, values in metrics.items():
        values_npy = np.asarray(values)
        aggregated_metrics[name] = {
            "mean": np.mean(values_npy),
            "std": np.std(values_npy)
        }
    return aggregated_metrics


def load_checkpoint(filepath: Path, model: nn.Module):
    print(f"Loading checkpoint from {filepath}")

    checkpoint = torch.load(filepath)
    model.model.load_state_dict(checkpoint["state_dict"])

    return model


def evaluate(folder: Path, downsampling_factor: int, distortion: Optional[str] = None, severity: int = 1):
    config_powerlines = powerlines_config()
    config_powerlines.data.batch_size.val = 1
    config_powerlines.minimal_logging = False
    config_powerlines.optimized_metrics = [  # all metrics that we want to be reported
        "cables/precision",
        "cables/recall",
        "cables/f1",
        "cables/correctness",
        "cables/completeness",
        "cables/quality",
        "poles/precision",
        "poles/recall",
        "poles/f1",
        "poles/correctness",
        "poles/completeness",
        "poles/quality"
    ]
    num_folds = 5

    metric_results = defaultdict(list)

    # Validation dataset
    for fold in range(num_folds):
        # Fold config
        fold_config = config_powerlines.copy()
        fold_config.data.cv.fold = fold

        if distortion is not None:
            fold_config.paths.complete_frames_root_folder = f"{config_powerlines.paths.complete_frames_root_folder}_{distortion}_{severity}"

        # Create data
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

        # Create and initialize model
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

        checkpoint_filepath = folder / f"fold_{fold}.pt"
        model = load_checkpoint(checkpoint_filepath, model)

        # Evaluate each for with given number of repetitions (to take distortions randomness into account)
        with torch.no_grad():
            metrics = validate(0, config, config_powerlines, {}, val_dataloader, model)

            for name, value in metrics.items():
                metric_results[name].append(value)

    return cross_validation_metrics(metric_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--downsampling-factor", type=int, default=16)
    parser.add_argument("--distortion", default=None)
    parser.add_argument("--severity", type=int)
    args = parser.parse_args()

    if args.distortion is None:
        print("Evaluating without any adverse weather distortions")
    else:
        print(f"Evaluating with distortion {args.distortion} @ {args.severity} severity")

    metrics = evaluate(Path(args.folder), args.downsampling_factor, args.distortion, args.severity)

    print("Metrics")
    for name, values in metrics.items():
        print(f"{name}: {values}")
