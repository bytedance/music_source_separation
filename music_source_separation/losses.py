import torch


def l1(output, target, **kwargs):
    return torch.mean(torch.abs(output - target))


def l1_wav(output, target, **kwargs):
    return l1(output, target)


def get_loss_function(loss_type):

    if loss_type == "l1_wav":
        return l1_wav

    else:
        raise NotImplementedError
