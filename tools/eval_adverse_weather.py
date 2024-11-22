import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from configs import config
from powerlines.data import seed
from powerlines.data.utils import POLES_WEIGHTS, CABLES_WEIGHTS
from tools import factory
from tools.train import powerlines_config
from utils.criterion import OhemCrossEntropy, CrossEntropy
from utils.function import validate
from utils.utils import FullModel


def cross_validation_metrics(metrics: Dict[str, List[float]], fold_sizes: List[int]):
    aggregated_metrics = {}
    w = np.asarray(fold_sizes)
    for name, values in metrics.items():
        x = np.asarray(values)
        mean_w = (x * w) / w.sum()
        m = np.count_nonzero(w)
        aggregated_metrics[name] = {
            "mean": mean_w,
            "std": np.sqrt(
                (w * ((x - mean_w) ** 2)).sum() /
                (((m - 1) / m) * w.sum())
            )
        }
    return aggregated_metrics


def load_checkpoint(filepath: Path, model: nn.Module):
    print(f"Loading checkpoint from {filepath}")

    checkpoint = torch.load(filepath)
    model.model.load_state_dict(checkpoint["state_dict"])

    return model


def evaluate(folder: Path, downsampling_factor: int, distortion: Optional[str] = None, severity: int = 1):
    config_powerlines = powerlines_config()
    config_powerlines.verbose = True
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
    fold_sizes = []

    # Validation dataset
    for fold in range(num_folds):
        # Fold config
        fold_config = config_powerlines.copy()
        fold_config.data.cv.fold = fold

        if distortion is not None:
            fold_config.paths.complete_frames = f"{fold_config.paths.complete_frames}_{distortion}_{severity}"

        # Create data
        val_dataset = factory.val_dataset(fold_config)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=fold_config.data.batch_size.val,
            shuffle=False,
            num_workers=fold_config.data.num_workers.val,
            pin_memory=False,
            persistent_workers=True,
            drop_last=False,
            worker_init_fn=seed.seed_worker,
            generator=seed.torch_generator()
        )

        # Create and initialize model
        ohem_config = fold_config.loss.ohem
        if ohem_config.enabled:
            print("Using OHEM in loss function")
            min_kept = int(ohem_config.keep_fraction * (fold_config.data.patch_size ** 2))
            semantic_seg_criterion = {
                "cables": OhemCrossEntropy(
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    thres=ohem_config.threshold,
                    min_kept=min_kept,
                    weight=torch.FloatTensor(CABLES_WEIGHTS).cuda()
                ),
                "poles": OhemCrossEntropy(
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    thres=ohem_config.threshold,
                    min_kept=min_kept,
                    weight=torch.FloatTensor(POLES_WEIGHTS).cuda()
                )
            }
        else:
            semantic_seg_criterion = {
                "cables": CrossEntropy(
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    weight=torch.FloatTensor(CABLES_WEIGHTS).cuda()
                ),
                "poles": CrossEntropy(
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    weight=torch.FloatTensor(POLES_WEIGHTS).cuda()
                )
            }
        pidnet = models.pidnet.get_seg_model(config)  # creates a pretrained model by default
        model = FullModel(pidnet, semantic_seg_criterion, poles_weight=fold_config.loss.poles_weight).cuda()

        checkpoint_filepath = folder / f"fold_{fold}.pt"
        model = load_checkpoint(checkpoint_filepath, model)

        # Evaluate each for with given number of repetitions (to take distortions randomness into account)
        with torch.no_grad():
            metrics = validate(fold_config, val_dataloader, model)

            for name, value in metrics.items():
                metric_results[name].append(value)

        fold_sizes.append(len(val_dataset))

    return cross_validation_metrics(metric_results, fold_sizes)


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
