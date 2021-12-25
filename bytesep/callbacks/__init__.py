from typing import List

import pytorch_lightning as pl
import torch.nn as nn


def get_callbacks(
    task_name: str,
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
) -> List[pl.Callback]:
    r"""Get callbacks of a task and config yaml file.

    Args:
        task_name: str
        config_yaml: str
        dataset_dir: str
        workspace: str, containing useful files such as audios for evaluation
        checkpoints_dir: str, directory to save checkpoints
        statistics_dir: str, directory to save statistics
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    if task_name == 'musdb18':

        from bytesep.callbacks.musdb18 import get_musdb18_callbacks

        return get_musdb18_callbacks(
            config_yaml=config_yaml,
            workspace=workspace,
            checkpoints_dir=checkpoints_dir,
            statistics_path=statistics_path,
            logger=logger,
            model=model,
            evaluate_device=evaluate_device,
        )

    elif task_name == 'voicebank-demand':

        from bytesep.callbacks.voicebank_demand import get_voicebank_demand_callbacks

        return get_voicebank_demand_callbacks(
            config_yaml=config_yaml,
            workspace=workspace,
            checkpoints_dir=checkpoints_dir,
            statistics_path=statistics_path,
            logger=logger,
            model=model,
            evaluate_device=evaluate_device,
        )

    elif task_name in [
        'vctk-musdb18',
        'violin-piano',
        'piano-symphony',
        'vctk-musdb18-audioset',
    ]:

        from bytesep.callbacks.instruments import get_instruments_callbacks

        return get_instruments_callbacks(
            config_yaml=config_yaml,
            workspace=workspace,
            checkpoints_dir=checkpoints_dir,
            statistics_path=statistics_path,
            logger=logger,
            model=model,
            evaluate_device=evaluate_device,
        )

    elif task_name in ['ambisonic-binaural']:

        from bytesep.callbacks.ambisonic_binaural import (
            get_ambisonic_binaural_callbacks,
        )

        return get_ambisonic_binaural_callbacks(
            config_yaml=config_yaml,
            workspace=workspace,
            checkpoints_dir=checkpoints_dir,
            statistics_path=statistics_path,
            logger=logger,
            model=model,
            evaluate_device=evaluate_device,
        )

    else:
        raise NotImplementedError
