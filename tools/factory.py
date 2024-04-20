import torch
from torch import nn


def optimizer(config, model: nn.Module):
    params_dict = dict(model.named_parameters())
    params = [{"params": list(params_dict.values()), "lr": config.TRAIN.LR}]

    if config.TRAIN.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV,
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        return torch.optim.Adam(
            params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WD
        )
    else:
        raise ValueError(f"Unsupported optimized {config.TRAIN.OPTIMIZER}")
