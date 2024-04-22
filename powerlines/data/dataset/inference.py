from typing import Optional

import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from powerlines.data.config import DataSourceConfig, LoadingConfig
from powerlines.data.utils import load_filtered_filepaths, load_annotations, load_complete_frame


class InferenceCablesDetectionDataset(BaseDataset):
    def __init__(
        self,
        data_source: DataSourceConfig,
        loading: LoadingConfig,
        num_frames: Optional[int] = None
    ):
        super().__init__(
            ignore_label=255,
            base_size=2048,
            crop_size=(3000, 4096),
            scale_factor=16
        )

        self.data_source = data_source
        self.loading = loading

        self.filepaths = load_filtered_filepaths(data_source)
        self.annotations = load_annotations(data_source)
        self.num_frames = num_frames if num_frames is not None else len(self.filepaths)

        self.cache = [{
            "timestamp": annotation.frame_timestamp(),
            "annotation": annotation
        } for annotation in self.annotations.values()]
        self.class_weights = torch.FloatTensor([1.0186, 54.7257]).cuda()

    def __getitem__(self, frame_id: int):
        frame = load_complete_frame(self.data_source, self.loading, self.cache[frame_id])
        name = str(frame["timestamp"])
        size = frame["image"].shape

        image = frame["image"]
        labels = frame["labels"]

        image, labels, edge = self.generate_sample(
            image, labels, generate_edge=False, multi_scale=False, is_flip=False, edge_pad=False
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

    def __len__(self) -> int:
        return self.num_frames
