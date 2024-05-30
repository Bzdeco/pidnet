from typing import Tuple, Dict

import torch
import torchvision.transforms.functional
from PIL import Image
from neptune import Run
from omegaconf import DictConfig

from datasets.base_dataset import IMAGENET_MEAN, IMAGENET_STD


def segmentation_as_image(segmentation_mask: torch.Tensor) -> torch.Tensor:
    dimensions = segmentation_mask.size()
    if len(dimensions) == 2:
        segmentation_mask = segmentation_mask.unsqueeze(0)

    num_channels = segmentation_mask.size(0)
    if num_channels == 1:
        smooth_segmentation_output = torch.clamp(segmentation_mask, min=0, max=1)
    elif num_channels == 2:
        smooth_segmentation_output = torch.softmax(segmentation_mask, dim=0)[1:, :]
    else:
        raise ValueError(f"Unexpected number of segmentation output dimensions: {num_channels}")

    smooth_segmentation_image = smooth_segmentation_output.expand(3, -1, -1).permute(1, 2, 0)
    return (smooth_segmentation_image * 255).int()


def upsample(image: torch.Tensor, factor: int) -> torch.Tensor:
    return torch.kron(image, torch.ones((factor, factor, 1))).to(image.dtype)


def pad_to(image: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    source_height, source_width = image.shape[:2]
    height, width = shape
    pad_height, pad_width = height - source_height, width - source_width
    assert pad_height >= 0 and pad_width >= 0, "Requested padding to smaller size than input image"

    return torch.nn.functional.pad(image, (0, 0, 0, pad_width, 0, pad_height), mode="constant", value=0).to(image.dtype)


imagenet_mean = torch.as_tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.as_tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(2)


def undo_image_preprocessing(image: torch.Tensor) -> torch.Tensor:
    return (255 * (image * imagenet_std + imagenet_mean)).int()


def _pred_target_vis(prediction: torch.Tensor, target: torch.Tensor, downsampling_factor: int) -> torch.Tensor:
    pred_vis = pad_to(upsample(segmentation_as_image(prediction.detach().cpu().float()), downsampling_factor), (3000, 4096))
    target_vis = pad_to(upsample(segmentation_as_image(target.detach().cpu().float()), downsampling_factor), (3000, 4096))
    return torch.concatenate((pred_vis, target_vis), dim=1)


def visualize_joint_detection(
    image: torch.Tensor,
    pred_cables: torch.Tensor,
    pred_poles: torch.Tensor,
    target_cables: torch.Tensor,
    target_poles: torch.Tensor,
    downsampling_factor: int
) -> Image:
    image_vis = undo_image_preprocessing(image.detach().cpu()).permute((1, 2, 0)).int()
    cables_vis = _pred_target_vis(pred_cables, target_cables, downsampling_factor)
    poles_vis = _pred_target_vis(pred_poles, target_poles, downsampling_factor)
    stacked_vis = torch.concatenate((image_vis, cables_vis, poles_vis))

    width, height = 1024 * 3, 750
    pil_image = torchvision.transforms.functional.to_pil_image(stacked_vis.permute((2, 0, 1)) / 255, mode="RGB")
    return pil_image.resize((width, height))


class VisualizationLogger:
    def __init__(self, run: Run, config: DictConfig):
        self._run = run
        visualization_config = config.visualization
        self._n_images_per_epoch = visualization_config.n_images_per_epoch
        self._every = visualization_config.every
        self._downsampling_factor = config.data.downsampling_factor

        self._n_logged = 0
        self._current_epoch = 0

    def visualize(
        self,
        epoch: int,
        images: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ):
        if epoch != self._current_epoch:
            self._current_epoch = epoch
            self._n_logged = 0

        if epoch % self._every == 0 and self._n_logged < self._n_images_per_epoch:
            for image, pred_cables, pred_poles, target_cables, target_poles in zip(
                images, predictions["cables"], predictions["poles"], targets["cables"], targets["poles"]
            ):
                if self._n_logged < self._n_images_per_epoch:
                    self._run["images"].append(
                        visualize_joint_detection(
                            image, pred_cables, pred_poles, target_cables, target_poles, self._downsampling_factor
                        ),
                        description=str(epoch)
                    )
                    self._n_logged += 1
