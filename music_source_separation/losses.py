from typing import Callable

import torch


def l1(output, target, **kwargs) -> torch.Tensor:
    return torch.mean(torch.abs(output - target))


def l1_wav(output, target, **kwargs) -> torch.Tensor:
    return l1(output, target)


def get_loss_function(loss_type) -> Callable:

    if loss_type == "l1_wav":
        return l1_wav

    else:
        raise NotImplementedError
