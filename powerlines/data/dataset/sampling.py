import random

from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from datasets.base_dataset import BaseDataset
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.utils import load_filtered_filepaths, sample_patch_center, load_annotations, \
    load_parameters_for_sampling, load_complete_frame
from powerlines.utils import parallelize


class TrainCablesDetectionDataset(BaseDataset):
    def __init__(
        self,
        data_source: DataSourceConfig,
        loading: LoadingConfig,
        sampling: SamplingConfig,
        num_frames: Optional[int] = None,
        num_workers: int = 4
    ):
        super().__init__(
            ignore_label=255,
            base_size=2048,
            crop_size=(1024, 1024),  # adapted to 1K square
            scale_factor=16
        )

        self.data_source = data_source
        self.loading = loading
        self.sampling = sampling

        self.filepaths = load_filtered_filepaths(data_source)
        self.annotations = load_annotations(data_source)
        self.num_frames = num_frames if num_frames is not None else len(self.filepaths)

        self.color_jitter = transforms.ColorJitter(
            brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.2, 0.2]
        )

        self._loading_data = self._frames_loading_data()
        self.cache = parallelize(
            load_parameters_for_sampling,
            self._loading_data,
            num_workers,
            f"Loading {data_source.data_source_subset} frames for configuration",
            use_threads=True
        )
        self.class_weights = torch.FloatTensor([1.0186, 54.7257]).cuda()
        self.sampling.configure_sampling(self.cache)

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
        frame = load_complete_frame(self.data_source, self.loading, self.cache[frame_id])
        size = frame["image"].shape
        name = str(frame["timestamp"])

        if self._should_sample_positive_sample(frame):
            patch_centers_data = frame["positive_sampling_centers_data"]
        else:
            patch_centers_data = frame["negative_sampling_centers_data"]

        y, x = sample_patch_center(patch_centers_data, self.sampling.non_sky_bias)
        image = self._extract_patch(frame["image"], y, x)
        labels = self._extract_patch(frame["labels"], y, x)

        image, labels, edge = self.generate_sample(image, labels, generate_edge=False)

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
