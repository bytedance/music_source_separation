import torch.nn as nn
import pytorch_lightning as pl

from music_source_separation.callbacks.musdb18_callbacks import get_musdb18_callbacks


def get_callbacks(
    task_name: str, 
    config_yaml: str, 
    workspace: str, 
    checkpoints_dir: str, 
    statistics_path: str, 
    logger: pl.loggers.TensorBoardLogger, 
    model: nn.Module, 
    evaluate_device: str
):

    if task_name == 'musdb18':
        return get_musdb18_callbacks(config_yaml, workspace, checkpoints_dir, statistics_path, logger, model, evaluate_device)

    else:
        raise NotImplementedError