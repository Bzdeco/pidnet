import math
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import precision_score, recall_score, average_precision_score

from powerlines.data.utils import pad_array_to_match_target_size, MAX_DISTANCE_MASK_VALUE

CCQConfusionMatrix = namedtuple("CCQConfusionMatrix", ["tp", "fp", "fn"])

INVALID_MASK_VALUE = np.iinfo(np.uint16).max


def downsample(array: np.ndarray, downsampling_factor: int, min_pooling: bool = False) -> np.ndarray:
    input_dtype = array.dtype
    downsampler = nn.MaxPool2d(
        kernel_size=(downsampling_factor, downsampling_factor),
        stride=(downsampling_factor, downsampling_factor)
    )

    inversion = -1 if min_pooling else 1
    array = inversion * array

    padded_distance_mask = pad_array_to_match_target_size(
        array[np.newaxis, :, :], downsampling_factor, padding_value=array.min()
    )
    pooled = downsampler(torch.tensor(padded_distance_mask.astype(float)).float())[0]

    return (inversion * pooled.detach().cpu().numpy()).astype(input_dtype)


def distance_transform(
    binary_mask: np.ndarray, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if np.count_nonzero(binary_mask) == 0:
        side_size = binary_mask.shape[-1]
        max_distance = int(side_size * math.sqrt(side_size))
        distance_mask = np.full_like(binary_mask, fill_value=max_distance, dtype=float)

        if return_indices:
            indices = np.zeros((2,) + binary_mask.shape, dtype=int)
            return distance_mask, indices
        else:
            return distance_mask

    outside_binary_mask = np.logical_not(binary_mask)
    return distance_transform_edt(outside_binary_mask, sampling=1, return_distances=True, return_indices=return_indices)


def relaxed_confusion_matrix(
    pred_prob: np.ndarray,
    target_distance_mask: np.ndarray,
    prob_thresholds: np.ndarray,
    tolerance_region: float
) -> CCQConfusionMatrix:
    num_thresh = len(prob_thresholds)
    # [num_thresholds, h, w]
    pred_bin_cable = np.repeat(pred_prob[np.newaxis], num_thresh, axis=0) >= prob_thresholds[:, np.newaxis, np.newaxis]
    gt_bin_cable = (target_distance_mask == 0)
    gt_distance = distance_transform(gt_bin_cable)

    # Compute distance mask from gt and predictions binarized with different thresholds
    pred_distance = np.zeros((num_thresh,) + pred_prob.shape[-2:])
    for i in range(num_thresh):
        pred_distance[i] = distance_transform(pred_bin_cable[i])

    # Compute confusion matrix entries
    valid_image_area = np.repeat((target_distance_mask != INVALID_MASK_VALUE)[np.newaxis, :, :], num_thresh, axis=0)
    true_pos_area = np.logical_and(
        np.repeat(gt_distance[np.newaxis, :, :] <= tolerance_region, num_thresh, axis=0), valid_image_area
    )
    false_pos_area = np.logical_and(np.logical_not(true_pos_area), valid_image_area)
    false_neg_area = np.logical_and(pred_distance > tolerance_region, valid_image_area)

    tp = np.logical_and(true_pos_area, pred_bin_cable).sum(axis=(1, 2))
    fp = np.logical_and(false_pos_area, pred_bin_cable).sum(axis=(1, 2))
    fn = np.logical_and(false_neg_area, gt_bin_cable).sum(axis=(1, 2))

    return CCQConfusionMatrix(tp=tp, fp=fp, fn=fn)


def visualize_distance_masks(pred_distance_mask: np.ndarray, target_distance_mask: np.ndarray, masked: bool):
    cmap = plt.get_cmap("gray").with_extremes(under="red", over="blue")
    combined = np.concatenate((pred_distance_mask, target_distance_mask), axis=1)

    folder = Path("/scratch/cvlab/home/gwizdala/output/distance_masks_temp")
    folder.mkdir(exist_ok=True)
    suffix = "_masked" if masked else ""
    filename = f"distance_mask_{len(list(folder.glob('*.png')))}{suffix}.png"

    print(np.count_nonzero(combined == INVALID_MASK_VALUE))
    combined[combined < 158] = 1 - np.clip(combined[combined < 158], a_min=0, a_max=128) / 128
    combined[combined >= 158] = 2
    plt.imshow(combined, cmap=cmap, vmin=0, vmax=1)
    plt.savefig(folder / filename)
    plt.close()


EPS = 1e-8


def correctness(tp: Union[int, np.ndarray], fp: Union[int, np.ndarray]) -> Union[float, np.ndarray]:  # precision
    return tp / (tp + fp + EPS)


def completeness(tp: Union[int, np.ndarray], fn: Union[int, np.ndarray]) -> Union[float, np.ndarray]:  # recall
    return tp / (tp + fn + EPS)


def quality(
    tp: Union[int, np.ndarray], fp: Union[int, np.ndarray], fn: Union[int, np.ndarray]
) -> Union[float, np.ndarray]:
    return tp / (tp + fp + fn + EPS)


def f1(precision: Union[float, np.ndarray], recall: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 2 * precision * recall / (precision + recall + EPS)


@dataclass
class SegmentationMetrics:
    def __init__(
        self, bin_thresholds: List[float], tolerance_region: float, minimal_logging: bool = False
    ):
        self._y_true = torch.empty((0,))
        self._y_pred_prob = torch.empty((0,))
        self._tolerance_region = tolerance_region

        self._bin_thresholds = np.asarray(bin_thresholds)
        num_bin_thresholds = len(self._bin_thresholds)
        self._ccq_tp = np.asarray([0] * num_bin_thresholds)
        self._ccq_fp = np.asarray([0] * num_bin_thresholds)
        self._ccq_fn = np.asarray([0] * num_bin_thresholds)

        self._minimal_logging = minimal_logging

    def __call__(self, prediction: torch.Tensor, target_labels: torch.Tensor):
        # Precision, recall, F1
        pred_probabilities = torch.softmax(prediction, dim=1)[:, 1].detach().cpu()
        flat_pred_probabilities = pred_probabilities.flatten()
        self._y_pred_prob = torch.cat((self._y_pred_prob, flat_pred_probabilities))
        gt_labels_flat = target_labels.flatten().detach().cpu()
        self._y_true = torch.cat((self._y_true, gt_labels_flat))

        # CCQ metrics
        for prediction, target in zip(pred_probabilities, target_labels):
            gt_distance = distance_transform((target == 1).detach().cpu().numpy())
            conf_matrix = relaxed_confusion_matrix(
                prediction.float().numpy(),
                gt_distance,
                self._bin_thresholds,
                self._tolerance_region
            )
            self._ccq_tp += conf_matrix.tp
            self._ccq_fp += conf_matrix.fp
            self._ccq_fn += conf_matrix.fn

    def compute(self) -> Dict[str, float]:
        # Reference metric – quality
        quality_values = quality(self._ccq_tp, self._ccq_fp, self._ccq_fn)
        best_quality_idx = np.argmax(quality_values)
        best_bin_threshold = self._bin_thresholds[best_quality_idx]

        # Results for best scoring threshold wrt reference quality metric
        results = {
            "quality": quality_values[best_quality_idx],
            "scoring_threshold": best_bin_threshold
        }

        # Results for other scoring thresholds
        if not self._minimal_logging:
            correctness_values = correctness(self._ccq_tp, self._ccq_fp)
            completeness_values = completeness(self._ccq_tp, self._ccq_fn)

            y_pred_prob = self._y_pred_prob.detach().cpu().numpy()
            y_pred = y_pred_prob >= best_bin_threshold
            y_true = self._y_true.detach().cpu().numpy()

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred_prob)

            results = {
                **results,
                "precision": precision,
                "recall": recall,
                "f1": f1(precision, recall),
                "average_precision": ap,
                "correctness": correctness_values[best_quality_idx],
                "completeness": completeness_values[best_quality_idx]
            }

            for i, bin_thresh in enumerate(self._bin_thresholds):
                y_pred = y_pred_prob >= bin_thresh
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)

                results[f"precision_{bin_thresh:.2f}"] = precision
                results[f"recall_{bin_thresh:.2f}"] = recall
                results[f"f1_{bin_thresh:.2f}"] = f1(precision, recall)
                results[f"correctness_{bin_thresh:.2f}"] = correctness_values[i]
                results[f"completeness_{bin_thresh:.2f}"] = completeness_values[i]
                results[f"quality_{bin_thresh:.2f}"] = quality_values[i]

        return results
