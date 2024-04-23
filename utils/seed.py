import random
from typing import Optional

import numpy as np
import torch

SEED = 304


def set_global_seeds(seed: Optional[int] = None):
    seed = seed or SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
