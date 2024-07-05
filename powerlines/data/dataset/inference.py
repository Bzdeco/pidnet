from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from datasets.base_dataset import BaseDataset
from powerlines.data.config import DataSourceConfig, LoadingConfig, SamplingConfig
from powerlines.data.utils import load_filtered_filepaths, load_annotations, load_complete_frame, CABLES_WEIGHTS, POLES_WEIGHTS


class InferenceCablesDetectionDataset(BaseDataset):
    def __init__(
        self,
        data_config: DictConfig,
        data_source: DataSourceConfig,
        loading: LoadingConfig,
        sampling: SamplingConfig,
        num_frames: Optional[int] = None,
        max_cells_away: int = 2
    ):
        super().__init__(
            ignore_label=255,
            base_size=data_config.patch_size,
            crop_size=(3000, 4096),
            scale_factor=data_config.augmentations.multi_scale.scale_factor
        )

        self.data_source = data_source
        self.loading = loading
        self.sampling = sampling
        self.max_cells_away = max_cells_away

        self.filepaths = load_filtered_filepaths(data_source)
        self.timestamps = list(map(lambda path: int(path.stem), self.filepaths))
        self.annotations = load_annotations(data_source)
        self.num_frames = num_frames if num_frames is not None else len(self.filepaths)

        self.cache = [{
            "timestamp": annotation.frame_timestamp(),
            "annotation": annotation
        } for annotation in self.annotations.values()]
        self.cables_class_weights = torch.FloatTensor(CABLES_WEIGHTS[self.max_cells_away]).cuda()
        self.poles_class_weights = torch.FloatTensor(POLES_WEIGHTS[self.max_cells_away]).cuda()

    def __getitem__(self, frame_id: int):
        annotation = self.annotations[self.timestamps[frame_id]]
        frame = load_complete_frame(annotation, self.data_source, self.sampling, self.loading, self.max_cells_away)
        timestamp = frame["timestamp"]
        name = str(timestamp)
        size = frame["image"].shape

        image = frame["image"]
        labels = {
            "cables": frame["labels_cables"],
            "poles": frame["labels_poles"]
        }

        image, labels, edge = self.generate_sample(image, labels, use_multi_scale=False, use_flipping=False)

        sample = {
            "image": image,
            "labels_cables": labels["cables"],
            "labels_poles": labels["poles"],
            "size": np.array(size),
            "name": name,
            "timestamp": timestamp
        }
        if edge is not None:
            sample["edge"] = edge

        return sample

    def __len__(self) -> int:
        return self.num_frames
