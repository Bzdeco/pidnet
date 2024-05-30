# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
from typing import Dict

import cv2
import numpy as np
import random

from torch.nn import functional as F
from torch.utils import data

y_k_size = 6
x_k_size = 6


def channel_last(image: np.ndarray) -> np.ndarray:
    if image.shape[0] <= 4 and all(map(lambda dim: dim > 4, image.shape[1:])):
        return np.transpose(image, (1, 2, 0))
    else:
        return image


def channel_first(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] <=4 and all(map(lambda dim: dim > 4, image.shape[:-1])):
        return np.transpose(image, (2, 0, 1))
    else:
        return image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BaseDataset(data.Dataset):
    def __init__(
        self,
        ignore_label=255,
        base_size=2048,
        crop_size=(512, 1024),
        scale_factor=16
    ):
        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor

        # ImageNet statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def input_transform(self, image: np.ndarray):
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, labels):
        return {
            entity: np.array(label).astype(np.uint8)
            for entity, label in labels.items()
        }

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(
                image.astype(float), 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue
            ).astype(image.dtype)

        return pad_image

    def rand_crop(self, image, labels, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.0, 0.0, 0.0))
        labels = {
            entity: self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))
            for entity, label in labels.items()
        }
        if edge is not None:
            edge = self.pad_image(edge, h, w, self.crop_size, (0.0,))

        new_h, new_w = image.shape[-2:]
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        labels = {
            entity: label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
            for entity, label in labels.items()
        }

        return image, labels, edge

    def multi_scale_aug(self, image, labels=None, edge=None, rand_scale=1.0, rand_crop=True):
        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]

        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

        # Channel last for resizing: https://stackoverflow.com/a/69348858
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        if labels is not None:
            labels = {
                entity: cv2.resize(label.astype(int), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                for entity, label in labels.items()
            }
        else:
            return image

        if rand_crop:
            image, labels, edge = self.rand_crop(image, labels, edge)

        return image, labels, edge

    def generate_sample(
        self,
        image: np.ndarray,
        labels: Dict[str, np.ndarray],
        use_multi_scale: bool = True,
        use_flipping: bool = True,
    ):
        edge = None
        image = channel_last(image)

        if use_multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, labels, edge = self.multi_scale_aug(image, labels, edge, rand_scale=rand_scale)

        image = channel_first(self.input_transform(image))
        labels = self.label_transform(labels)

        # Horizontal flipping, copy to fix negative strides
        if use_flipping:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip].copy()
            labels = {
                entity: label[:, ::flip].copy()
                for entity, label in labels.items()
            }

        return image, labels, edge

    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]
        
        pred = F.interpolate(input=pred, size=size[-2:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        return pred.exp()
