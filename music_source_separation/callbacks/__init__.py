import torch.nn as nn
import pytorch_lightning as pl

from music_source_separation.callbacks.musdb18_callbacks import get_musdb18_callbacks
from music_source_separation.callbacks.voicebank_demand_callbacks import (
    get_voicebank_demand_callbacks,
)


def get_callbacks(
    task_name: str,
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
):
    r"""Get callbacks of a task and config yaml file."""
    if task_name == 'musdb18':
        return get_musdb18_callbacks(
            config_yaml,
            workspace,
            checkpoints_dir,
            statistics_path,
            logger,
            model,
            evaluate_device,
        )

    elif task_name == 'voicebank-demand':
        return get_voicebank_demand_callbacks(
            config_yaml,
            workspace,
            checkpoints_dir,
            statistics_path,
            logger,
            model,
            evaluate_device,
        )

    else:
        raise NotImplementedError
