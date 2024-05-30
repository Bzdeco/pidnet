# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import re
import time
from pathlib import Path
from typing import Optional, Dict

import neptune
import numpy as np

import torch
import torch.nn as nn
from neptune import Run
from omegaconf import DictConfig
from torch import ModuleDict


class FullModel(nn.Module):
    def __init__(self, model: nn.Module, semantic_seg_losses: ModuleDict, poles_weight: float = 1.0):
        super(FullModel, self).__init__()
        self.model = model
        self.sem_seg_losses = semantic_seg_losses

        self.entity_to_channel = {"cables": 0, "poles": 1}
        self.weights = {"cables": 1.0, "poles": poles_weight}

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs: torch.Tensor, labels: Dict[str, torch.Tensor], *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)

        loss = None
        accuracies = {}
        predictions = {}
        for entity, sem_seg_loss in self.sem_seg_losses.items():
            channel = self.entity_to_channel[entity]
            weight = self.weights[entity]
            dense_pred_segmentation = outputs["p"][..., channel, :, :]
            coarse_pred_segmentation = outputs["main"][..., channel, :, :]

            accuracies[entity] = self.pixel_acc(dense_pred_segmentation, labels[entity])
            seg_loss = weight * sem_seg_loss([dense_pred_segmentation, coarse_pred_segmentation], labels[entity])

            if loss is None:
                loss = seg_loss
            else:
                loss += seg_loss

            predictions[entity] = {"p": dense_pred_segmentation, "main": coarse_pred_segmentation}

        return torch.unsqueeze(loss, 0), predictions, accuracies


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def checkpoint_folder(config: DictConfig, run_id: int) -> Path:
    folder = Path(config.paths.checkpoints) / str(run_id)
    folder.mkdir(exist_ok=True)
    return folder


def create_neptune_run(name: str, resume: bool = False, from_run_id: Optional[int] = None) -> Run:
    # Attach to the existing run if loading from checkpoint and resuming the run
    with_id = f"POW-{from_run_id}" if from_run_id is not None and resume else None
    run_name = name if with_id is None else None

    return neptune.init_run(
        project="jakubg/powerlines",
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        with_id=with_id,
        name=run_name,
        mode="async",
        capture_stdout=True,
        capture_stderr=True,
        capture_traceback=True,
        capture_hardware_metrics=True,
        flush_period=300,
        source_files=[]  # do not log source code
    )


neptune_id_pattern = re.compile(r"\w+-(?P<id>\d+)")


def run_id(run: neptune.Run) -> int:
    neptune_id = run["sys/id"].fetch()
    match = neptune_id_pattern.match(neptune_id)
    return int(match.group("id"))


def create_logger(cfg_name, output_folder: Path, phase='train'):
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = output_folder / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = output_folder / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calculate the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    optimizer.param_groups[0]["lr"] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr
