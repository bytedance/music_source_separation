from typing import List

import pytorch_lightning as pl
import torch.nn as nn

from bytesep.callbacks.musdb18_callbacks import get_musdb18_callbacks
# from bytesep.callbacks.voicebank_demand_callbacks import (
#     get_voicebank_demand_callbacks,
# )


def get_callbacks(
    task_name: str,
    config_yaml: str,
    dataset_dir: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
) -> List[pl.Callback]:
    r"""Get callbacks of a task and config yaml file.

    Args:
        config_yaml: str
        dataset_dir: str
        workspace: str
        checkpoints_dir: str
        statistics_dir: str
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    if task_name == 'musdb18':
        return get_musdb18_callbacks(
            config_yaml=config_yaml,
            dataset_dir=dataset_dir,
            workspace=workspace,
            checkpoints_dir=checkpoints_dir,
            statistics_path=statistics_path,
            logger=logger,
            model=model,
            evaluate_device=evaluate_device,
        )

    elif task_name == 'voicebank-demand':
        return get_voicebank_demand_callbacks(
            config_yaml=config_yaml,
            dataset_dir=dataset_dir,
            workspace=workspace,
            checkpoints_dir=checkpoints_dir,
            statistics_path=statistics_path,
            logger=logger,
            model=model,
            evaluate_device=evaluate_device,
        )

    else:
        raise NotImplementedError
