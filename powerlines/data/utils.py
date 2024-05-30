import math
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union

import numpy as np
import torch
from torch import nn

from powerlines.data.annotations import ImageAnnotations, num_cable_samples_in_frame, num_pole_samples_in_frame
from powerlines.data.config import DataSourceConfig, SamplingConfig, LoadingConfig, _filter_excluded_filepaths, \
    _filter_selected_filepaths
from powerlines.utils import load_npy, load_image

INVALID_MASK_VALUE = np.iinfo(np.uint16).max
CABLES_WEIGHTS = [1.0186, 54.7257]
POLES_WEIGHTS = [1.0297, 34.6541]


def load_filepaths(data_source_config) -> List[Path]:
    return sorted(list(filter(
        lambda path: data_source_config.belongs_to_subset_frames(path),
        data_source_config.inputs_folder.glob("*")
    )))


def load_filtered_filepaths(data_source_config) -> List[Path]:
    return _filter_excluded_filepaths(
        _filter_selected_filepaths(
            load_filepaths(data_source_config),
            data_source_config.selected_timestamps
        ), data_source_config.excluded_timestamps
    )


def positive_sampled_patch_centers_data(
    frame_annotations: ImageAnnotations,
    distance_mask: np.ndarray,
    perturbation_size: int,
    patch_size: int,
    remove_excluded_area: bool
) -> Dict[str, np.ndarray]:
    sampling_within_distance_area = distance_mask[0, :, :] <= perturbation_size
    return _sampled_patch_centers(
        sampling_within_distance_area, patch_size, remove_excluded_area, frame_annotations
    )


def negative_sampled_patch_centers_data(
    frame_annotations: ImageAnnotations,
    distance_mask: np.ndarray,
    patch_size: int,
    remove_excluded_area: bool
) -> Dict[str, np.ndarray]:
    sampling_outside_distance_area = np.logical_and(
        distance_mask[0, :, :] > (patch_size / math.sqrt(2)),
        distance_mask[0, :, :] != INVALID_MASK_VALUE
    )
    return _sampled_patch_centers(
        sampling_outside_distance_area, patch_size, remove_excluded_area, frame_annotations
    )


def _sampled_patch_centers(
    sampling_area: np.ndarray,
    patch_size: int,
    remove_excluded_area: bool,
    frame_annotations: ImageAnnotations
) -> Dict[str, np.ndarray]:
    removed_margins = remove_image_margins(sampling_area, patch_size)

    if remove_excluded_area:
        final_sampling_area = remove_exclusion_area_from_mask(
            removed_margins, frame_annotations, perturbation_size=0, patch_size=patch_size
        )
    else:
        final_sampling_area = removed_margins

    patch_centers = patch_centers_from_sampling_area(final_sampling_area)
    ys = patch_centers[:, 0]
    non_sky_bias = int(sampling_area.shape[0] / 2)
    ground_region = (ys > non_sky_bias)
    sky_region = ~ground_region

    return {
        "has_ground_region": np.any(ground_region),
        "ground": patch_centers[ground_region],
        "sky": patch_centers[sky_region]
    }


def patch_centers_from_sampling_area(sampling_area: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(sampling_area)
    return np.stack((ys, xs), axis=1)


def remove_image_margins(sampling_area_mask: np.ndarray, patch_size: int) -> np.ndarray:
    inner_area = np.zeros_like(sampling_area_mask)
    half_patch_size = patch_size // 2
    inner_area[half_patch_size:-half_patch_size, half_patch_size:-half_patch_size] = True
    return np.logical_and(sampling_area_mask, inner_area)


def remove_exclusion_area_from_mask(
    mask: np.ndarray, image_annotations: ImageAnnotations, perturbation_size: int, patch_size: int
) -> np.ndarray:
    exclusion_mask = exclusion_area_mask(image_annotations, perturbation_size, patch_size, mask.shape)
    return np.logical_and(mask, np.logical_not(exclusion_mask))


def exclusion_area_mask(
    image_annotations: ImageAnnotations, perturbation_size: int, patch_size: int, shape: Tuple[int, int]
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width))
    margin = perturbation_size + patch_size // 2

    for exclusion_zone in image_annotations.exclusion_zones:
        top, left = exclusion_zone.top_left
        bottom, right = exclusion_zone.bottom_right
        mask[
            max(0, int(top - margin)):min(height, int(bottom + margin + 1)),
            max(0, int(left - margin)):min(width, int(right + margin + 1))
        ] = True

    return mask


def sample_patch_center(
    patch_centers_data: Dict[str, np.ndarray], non_sky_bias: Optional[float] = None
) -> np.ndarray:
    if non_sky_bias is not None and random.random() <= non_sky_bias and patch_centers_data["has_ground_region"]:
        used_patch_centers = patch_centers_data["ground"]
    else:
        used_patch_centers = np.concatenate([patch_centers_data["sky"], patch_centers_data["ground"]])

    # Sample patch center
    patch_center_idx = random.randint(0, len(used_patch_centers) - 1)

    return used_patch_centers[patch_center_idx, :]


MAX_DISTANCE_MASK_VALUE = 128


def load_annotations(data_source_config: DataSourceConfig) -> Dict[int, Any]:
    fold_annotations = {}

    for frame_annotations in data_source_config.annotations():
        frame_timestamp = frame_annotations.frame_timestamp()
        if frame_timestamp in data_source_config.timestamps:
            fold_annotations[frame_timestamp] = frame_annotations

    return fold_annotations


def load_distance_masks(
    data_source_config: DataSourceConfig, loading_config: LoadingConfig, frame_timestamp: int
) -> Dict[str, Optional[np.ndarray]]:
    filename = f"{frame_timestamp}.npy"

    cables_distance_mask = load_npy(
        data_source_config.distance_masks_folder / filename
    ) if loading_config.distance_mask else None

    poles_distance_mask = load_npy(
        data_source_config.poles_distance_masks_folder / filename
    ) if loading_config.poles_distance_mask else None

    exclusion_zones_distance_mask = load_npy(
        data_source_config.exclusion_zones_distance_masks_folder / filename
    ) if loading_config.exclusion_area_mask else None

    return {
        "distance_mask": cables_distance_mask,
        "poles_distance_mask": poles_distance_mask,
        "exclusion_zones_distance_mask": exclusion_zones_distance_mask
    }


def load_processed_label_image(filepath: Path) -> np.ndarray:
    return load_image(filepath)[0] > 0


def load_parameters_for_sampling(loaded_frame_data: Dict[str, Any]) -> Dict[str, Any]:
    data_source: DataSourceConfig = loaded_frame_data["data_source"]
    sampling: SamplingConfig = loaded_frame_data["sampling"]
    loading: LoadingConfig = loaded_frame_data["loading"]
    timestamp = loaded_frame_data["timestamp"]
    annotation = loaded_frame_data["annotation"]
    distance_masks = load_distance_masks(data_source, loading, timestamp)
    return frame_parameters(annotation, sampling, distance_masks)


def frame_parameters(
    annotation: ImageAnnotations,
    sampling: SamplingConfig,
    distance_masks: Dict[str, Any]
):
    # Sample relative to cables and poles combined in a single distance mask
    sampling_source_distance_mask = np.minimum(distance_masks["distance_mask"], distance_masks["poles_distance_mask"])
    num_positive_samples = max(
        num_cable_samples_in_frame(annotation.cables("visible"), sampling.patch_size),
        num_pole_samples_in_frame(annotation)
    )

    positive_patch_centers_data = positive_sampled_patch_centers_data(
        annotation,
        sampling_source_distance_mask,
        sampling.perturbation_size,
        sampling.patch_size,
        sampling.remove_excluded_area
    )
    negative_patch_centers_data = negative_sampled_patch_centers_data(
        annotation,
        sampling_source_distance_mask,
        sampling.patch_size,
        sampling.remove_excluded_area
    )

    return {
        "timestamp": annotation.frame_timestamp(),
        "annotation": annotation,
        "positive_sampling_centers_data": positive_patch_centers_data,
        "negative_sampling_centers_data": negative_patch_centers_data,
        "has_positive_samples": len(positive_patch_centers_data["ground"]) + len(positive_patch_centers_data["sky"]) > 0,
        "has_negative_samples": len(negative_patch_centers_data["ground"]) + len(negative_patch_centers_data["sky"]) > 0,
        "num_positive_samples": num_positive_samples
    }


def load_complete_frame(
    annotation: ImageAnnotations,
    data_source: DataSourceConfig,
    sampling: SamplingConfig,
    loading: LoadingConfig
) -> Dict[str, Any]:
    timestamp = annotation.frame_timestamp()
    frame_image = load_npy(data_source.input_filepath(timestamp))
    distance_masks = load_distance_masks(data_source, loading, timestamp)
    labels_cables = load_processed_label_image(data_source.labels_folder / f"{timestamp}.png") if loading.labels else None
    poles_distance_mask = distance_masks["poles_distance_mask"]

    return {
        "timestamp": timestamp,
        "image": frame_image,
        "labels_cables": labels_cables,
        "labels_poles": nevbw_labels_from_distance_mask(poles_distance_mask[0]),
        "distance_mask": distance_masks["distance_mask"],
        "poles_distance_mask": poles_distance_mask,
        "exclusion_zones_distance_mask": distance_masks["exclusion_zones_distance_mask"],
        **frame_parameters(annotation, sampling, distance_masks)
    }


def train_augmentations():
    raise NotImplementedError


def evaluation_augmentations():
    raise NotImplementedError


def inference_augmentations():
    raise NotImplementedError


def compute_extended_frame_padding(image_size: Tuple[int, ...], downsampling_factor: int):
    height, width = image_size[-2:]

    # Pad image so that it corresponds in shape to min. image step size
    incomplete_bottom = height - (height // downsampling_factor) * downsampling_factor
    incomplete_right = width - (width // downsampling_factor) * downsampling_factor
    pad_bottom = downsampling_factor - incomplete_bottom if incomplete_bottom > 0 else 0
    pad_right = downsampling_factor - incomplete_right if incomplete_right > 0 else 0

    return pad_bottom, pad_right


def num_side_patches(side_size: int, patch_size: int, step_size: int) -> int:
    return int(math.ceil((side_size - patch_size) / step_size)) + 1


def pad_array_to_match_target_size(array: np.ndarray, downsampling_factor: int, padding_value: float) -> np.ndarray:
    # Pads image so that each targets pixel corresponds to a complete and identical area in the image
    pad_bottom, pad_right = compute_extended_frame_padding(array.shape, downsampling_factor)
    return np.pad(array, ((0, 0), (0, pad_bottom), (0, pad_right)), mode="constant", constant_values=padding_value)


def pad_tensor_to_match_target_size(tensor: torch.Tensor, downsampling_factor: int, padding_value: float) -> torch.Tensor:
    pad_bottom, pad_right = compute_extended_frame_padding(tensor.shape, downsampling_factor)
    return torch.nn.functional.pad(tensor, (0, pad_right, 0, pad_bottom), mode="constant", value=padding_value)


def cut_into_complete_set_of_patches(
    image: Union[torch.Tensor, np.ndarray], patch_size: int, step_size: int
) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]]]:
    assert len(image.shape) == 3, f"Expected image as (C, H, W), got {len(image.shape)} dimensions"
    height, width = image.shape[1:]
    is_array = isinstance(image, np.ndarray)

    patches = []
    shifts = []
    for i in range(num_side_patches(height, patch_size, step_size)):
        for j in range(num_side_patches(width, patch_size, step_size)):
            y = i * step_size
            x = j * step_size
            if y + patch_size > height:
                y = height - patch_size
            if x + patch_size > width:
                x = width - patch_size

            patches.append(image[:, y:(y + patch_size), x:(x + patch_size)])
            if is_array:
                shifts.append(np.asarray([x, y]))
            else:
                shifts.append(torch.as_tensor([x, y]))

    return patches, shifts


_downsampler = nn.MaxPool2d(kernel_size=2, stride=2).cuda()


def downsample_labels(labels: torch.Tensor, grid_size: int, adjust_to_divisible: bool) -> torch.Tensor:
    if adjust_to_divisible:
        input_labels = pad_tensor_to_match_target_size(labels, grid_size, padding_value=0)
    else:
        input_labels = labels

    batch_size, height, width = input_labels.size()
    half_grid_size = grid_size // 2

    x_steps = width // grid_size
    base_xs = torch.linspace(half_grid_size - 1, half_grid_size - 1 + (x_steps - 1) * grid_size, x_steps)
    xs = base_xs.repeat_interleave(2)
    xs[1::2] += 1
    y_steps = height // grid_size
    base_ys = torch.linspace(half_grid_size - 1, half_grid_size - 1 + (y_steps - 1) * grid_size, y_steps)
    ys = base_ys.repeat_interleave(2)
    ys[1::2] += 1
    xs_indices, ys_indices = torch.meshgrid(xs, ys, indexing="xy")

    grid_centers = input_labels[:, ys_indices.long().flatten(), xs_indices.long().flatten()].reshape(
        (batch_size, height // grid_size * 2, width // grid_size * 2)
    )
    return _downsampler(grid_centers.unsqueeze(1)).squeeze(1)


GRID_SIZE = 16
NEVBW_MAX_CELLS_AWAY = 2


def nevbw_labels_from_distance_mask(
    distance_mask: np.ndarray, grid_size: int = GRID_SIZE, max_cells_away: int = NEVBW_MAX_CELLS_AWAY
) -> np.ndarray:
    """
    Method used to obtain labels from distance mask based on the method description in Stambler et al. paper
    """

    return distance_mask <= max_cells_away * grid_size  # within max_cells_away grid cells distance
