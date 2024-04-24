import random

from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from omegaconf import DictConfig

from datasets.base_dataset import BaseDataset
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.utils import load_filtered_filepaths, sample_patch_center, load_annotations, \
    load_parameters_for_sampling, load_complete_frame
from powerlines.utils import parallelize


class TrainCablesDetectionDataset(BaseDataset):
    def __init__(
        self,
        data_config: DictConfig,
        data_source: DataSourceConfig,
        loading: LoadingConfig,
        sampling: SamplingConfig,
        num_frames: Optional[int] = None,
        num_workers: int = 4
    ):
        super().__init__(
            ignore_label=255,
            base_size=data_config.patch_size,
            crop_size=(data_config.patch_size,) * 2,
            scale_factor=data_config.augmentations.multi_scale.scale_factor
        )

        self.data_source = data_source
        self.loading = loading
        self.sampling = sampling

        self.filepaths = load_filtered_filepaths(data_source)
        self.timestamps = list(map(lambda path: int(path.stem), self.filepaths))
        self.annotations = load_annotations(data_source)
        self.num_frames = num_frames if num_frames is not None else len(self.filepaths)

        # Data augmentations
        self.use_color_jitter = data_config.augmentations.color_jitter.enabled
        self.use_flipping = data_config.augmentations.flip.enabled
        self.use_multi_scale = data_config.augmentations.multi_scale.enabled

        magnitude = data_config.augmentations.color_jitter.magnitude
        self.color_jitter = transforms.ColorJitter(
            brightness=[1 - magnitude, 1 + magnitude],
            contrast=[1 - magnitude, 1 + magnitude],
            saturation=[1 - magnitude, 1 + magnitude],
            hue=[-magnitude, magnitude]
        )

        self._loading_data = self._frames_loading_data()
        parameters = parallelize(
            load_parameters_for_sampling,
            self._loading_data,
            num_workers,
            f"Loading {data_source.data_source_subset} frames for configuration",
            use_threads=True
        )
        self.class_weights = torch.FloatTensor([1.0186, 54.7257]).cuda()
        self.sampling.configure_sampling(parameters)
        del parameters

    def _frames_loading_data(self) -> List[Dict[str, Any]]:
        return [self._single_frame_loading_data(frame_id) for frame_id in range(self.num_frames)]

    def _single_frame_loading_data(self, frame_id: int) -> Dict[str, Any]:
        filepath = self.filepaths[frame_id]
        timestamp = int(filepath.stem)
        return {
            "data_source": self.data_source,
            "loading": self.loading,
            "sampling": self.sampling,
            "timestamp": timestamp,
            "annotation": self.annotations[timestamp],
        }

    def __getitem__(self, idx: int):
        frame_id = self.sampling.frame_idx_for_sample(idx)
        annotation = self.annotations[self.timestamps[frame_id]]
        frame = load_complete_frame(annotation, self.data_source, self.sampling, self.loading)

        size = frame["image"].shape
        name = str(frame["timestamp"])

        if self._should_sample_positive_sample(frame):
            patch_centers_data = frame["positive_sampling_centers_data"]
        else:
            patch_centers_data = frame["negative_sampling_centers_data"]

        y, x = sample_patch_center(patch_centers_data, self.sampling.non_sky_bias)
        image = self._extract_patch(frame["image"], y, x)
        labels = self._extract_patch(frame["labels"], y, x)

        image, labels, edge = self.generate_sample(
            image, labels, generate_edge=False, use_multi_scale=self.use_multi_scale, use_flipping=self.use_flipping
        )

        sample = {
            "image": image,
            "labels": labels,
            "size": np.array(size),
            "name": name
        }
        if edge is not None:
            sample["edge"] = edge

        return sample

    def input_transform(self, image: np.ndarray):
        # Rescale
        image = image.astype(np.float32)
        image = image / 255.0

        # Color jitter
        if self.use_color_jitter:
            image = self.color_jitter(torch.from_numpy(image).permute((2, 0, 1))).permute((1, 2, 0)).numpy()

        # Normalize
        image -= self.mean
        image /= self.std

        return image

    def _extract_patch(self, input: Optional[np.ndarray], y: int, x: int) -> Optional[np.ndarray]:
        if input is None:
            return None

        return input[
            ...,
            (y - self.sampling.half_patch_size):(y + self.sampling.half_patch_size),
            (x - self.sampling.half_patch_size):(x + self.sampling.half_patch_size)
        ]

    def _should_sample_positive_sample(self, cached_frame: Dict[str, Any]):
        should_sample_positive = random.random() <= self.sampling.positive_sample_prob
        has_positive_samples = cached_frame["has_positive_samples"]
        has_negative_samples = cached_frame["has_negative_samples"]

        return (should_sample_positive and has_positive_samples) or not has_negative_samples

    def __len__(self):
        return self.sampling.num_samples
