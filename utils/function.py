# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os

import numpy as np
from neptune import Run
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from torch.nn import functional as F

from powerlines.data.utils import downsample_labels
from powerlines.evaluation import segmentation_metrics
from powerlines.visualization import VisualizationLogger
from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate


def train(
    run: Run,
    config_powerlines: DictConfig,
    epoch: int,
    epoch_iters: int,
    num_iters: int,
    dataloader: DataLoader,
    optimizer,
    model
):
    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    cur_iters = epoch * epoch_iters

    for i_iter, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch["image"].cuda()
        bd_gts = batch["edge"].float().cuda() if "edge" in batch else None

        labels = batch["labels"].cuda()
        labels = downsample_labels(labels.float(), grid_size=16, adjust_to_divisible=False).long()

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # update average loss
        loss_meter.update(loss.item())
        accuracy_meter.update(acc.item())

        optimizer_config = config_powerlines.optimizer
        if optimizer_config.adjust_lr:
            adjust_learning_rate(optimizer, optimizer_config.lr, num_iters, i_iter + cur_iters)

    run["metrics/train/loss/total"].log(loss_meter.average())
    run["metrics/train/accuracy"].log(accuracy_meter.average())


def validate(
    epoch: int, config, config_powerlines: DictConfig, run: Run, dataloader: DataLoader, model: nn.Module
) -> float:
    model.eval()

    loss_meter = AverageMeter()
    seg_metrics = segmentation_metrics()
    vis_logger = VisualizationLogger(run, config_powerlines)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            images = batch["image"].cuda()
            bd_gts = batch["edge"].float().cuda() if "edge" in batch else None
            labels = batch["labels"].cuda()
            labels = downsample_labels(labels.float(), grid_size=16, adjust_to_divisible=False).long()

            # Inference
            losses, predictions, _, _ = model(images, labels, bd_gts)

            # Update metrics
            seg_predictions = predictions[config.TEST.OUTPUT_INDEX]
            seg_metrics(seg_predictions, labels)
            loss_meter.update(losses.mean().item())
            vis_logger.visualize(epoch, images, seg_predictions, labels)

    # Log metrics
    run["metrics/val/loss/total"].log(loss_meter.average())
    metrics = seg_metrics.compute()
    for name, value in metrics.items():
        run[f"metrics/val/{name}"].log(value)

    return metrics[config_powerlines.optimized_metric]


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
