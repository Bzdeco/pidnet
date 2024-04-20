import concurrent.futures
from pathlib import Path
from typing import Any, Dict, Iterable, Callable, Optional

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


def load_yaml(filepath: Path) -> Dict[str, Any]:
    with filepath.open() as file:
        return yaml.safe_load(file)


def load_npy(filepath: Path) -> np.ndarray:
    return np.load(str(filepath))


def load_image(image_path: Path) -> np.ndarray:
    return pillow_to_numpy(load_pillow_image(image_path))


def pillow_to_numpy(image: Image) -> np.ndarray:
    image_arr = np.array(image)
    if len(image_arr.shape) == 3:
        return np.transpose(image_arr, (2, 0, 1))
    else:
        return image_arr


def load_pillow_image(image_path: Path) -> Image:
    return Image.open(image_path)


def parallelize(
    function: Callable, data: Iterable, num_workers: int, description: Optional[str] = None, use_threads: bool = False
):
    concurrent_executor = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
    # Using executor, additionally with tqdm: https://stackoverflow.com/a/52242947
    with concurrent_executor(max_workers=num_workers) as executor:
        return list(tqdm(executor.map(function, data), total=len(list(data)), desc=description))
