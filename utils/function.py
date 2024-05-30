# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
from typing import List

import numpy as np
from neptune import Run
from omegaconf import DictConfig
from torch import nn
from torch.cuda.amp import GradScaler
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
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    model: nn.Module
):
    model.train()
    torch.cuda.empty_cache()

    loss_meter = AverageMeter()
    accuracy_meters = {"cables": AverageMeter(), "poles": AverageMeter()}
    cur_iters = epoch * epoch_iters

    iterator = tqdm(dataloader, desc="Training") if config_powerlines.verbose else dataloader
    for i_iter, batch in enumerate(iterator):
        images = batch["image"].cuda()
        labels = {
            "cables": downsample_labels(batch["labels_cables"].cuda().float(), grid_size=16, adjust_to_divisible=False).long(),
            "poles": downsample_labels(batch["labels_poles"].cuda().float(), grid_size=16, adjust_to_divisible=False).long()
        }

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True, cache_enabled=True):
            losses, _, accuracies = model(images, labels)
            loss = losses.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # update average loss
        loss_meter.update(loss.item())
        for entity, acc in accuracies.items():
            accuracy_meters[entity].update(acc.item())

        optimizer_config = config_powerlines.optimizer
        if optimizer_config.adjust_lr:
            adjust_learning_rate(optimizer, optimizer_config.lr, num_iters, i_iter + cur_iters)

        del images, labels, losses, accuracies

    run["metrics/train/loss/total"].append(loss_meter.average(), step=epoch)
    run["metrics/train/accuracy/cables"].append(accuracy_meters["cables"].average(), step=epoch)
    run["metrics/train/accuracy/poles"].append(accuracy_meters["poles"].average(), step=epoch)


def validate(
    epoch: int, config, config_powerlines: DictConfig, run: Run, dataloader: DataLoader, model: nn.Module
) -> List[float]:
    model.eval()
    torch.cuda.empty_cache()

    loss_meter = AverageMeter()
    seg_metrics = {
        "cables": segmentation_metrics(),
        "poles": segmentation_metrics()
    }
    vis_logger = VisualizationLogger(run, config_powerlines)

    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Validating") if config_powerlines.verbose else dataloader
        for idx, batch in enumerate(iterator):
            images = batch["image"].cuda()
            labels = {
                "cables": downsample_labels(
                    batch["labels_cables"].cuda().float(), grid_size=16, adjust_to_divisible=False
                ).long(),
                "poles": downsample_labels(
                    batch["labels_poles"].cuda().float(), grid_size=16, adjust_to_divisible=False
                ).long()
            }

            # Inference
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True, cache_enabled=True):
                losses, predictions, _ = model(images, labels)

            # Update metrics
            vis_predictions = {}
            for entity, seg_metric in seg_metrics.items():
                seg_prediction = predictions[entity]["main"]
                vis_predictions[entity] = seg_prediction
                seg_metric(seg_prediction, labels[entity])

            loss_meter.update(losses.mean().item())
            vis_logger.visualize(epoch, images, vis_predictions, labels)

    # Log metrics
    run["metrics/val/loss/total"].append(loss_meter.average(), step=epoch)
    all_metrics = {}
    for entity, seg_metric in seg_metrics.items():
        metrics = seg_metric.compute()
        for name, value in metrics.items():
            run[f"metrics/val/{entity}/{name}"].append(value, step=epoch)
            all_metrics[f"{entity}/{name}"] = value

    return [all_metrics[metric_name] for metric_name in config_powerlines.optimized_metrics]


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
