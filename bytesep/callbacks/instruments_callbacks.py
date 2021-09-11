import logging
import os
import time
from typing import List, NoReturn

import librosa
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

from bytesep.callbacks.base_callbacks import SaveCheckpointsCallback
from bytesep.inference import Separator
from bytesep.utils import StatisticsContainer, calculate_sdr, read_yaml


def get_instruments_callbacks(
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
) -> List[pl.Callback]:
    """Get Voicebank-Demand callbacks of a config yaml.

    Args:
        config_yaml: str
        workspace: str
        checkpoints_dir: str, directory to save checkpoints
        statistics_dir: str, directory to save statistics
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    configs = read_yaml(config_yaml)
    task_name = configs['task_name']
    target_source_types = configs['train']['target_source_types']
    input_channels = configs['train']['channels']
    mono = True if input_channels == 1 else False
    test_audios_dir = os.path.join(workspace, "evaluation_audios", task_name, "test")
    sample_rate = configs['train']['sample_rate']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    test_batch_size = configs['evaluate']['batch_size']
    test_segment_seconds = configs['evaluate']['segment_seconds']

    test_segment_samples = int(test_segment_seconds * sample_rate)
    assert len(target_source_types) == 1
    target_source_type = target_source_types[0]

    # save checkpoint callback
    save_checkpoints_callback = SaveCheckpointsCallback(
        model=model,
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    # statistics container
    statistics_container = StatisticsContainer(statistics_path)

    # evaluation callback
    evaluate_test_callback = EvaluationCallback(
        model=model,
        target_source_type=target_source_type,
        input_channels=input_channels,
        sample_rate=sample_rate,
        mono=mono,
        evaluation_audios_dir=test_audios_dir,
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    callbacks = [save_checkpoints_callback, evaluate_test_callback]
    # callbacks = [save_checkpoints_callback]

    return callbacks


class EvaluationCallback(pl.Callback):
    def __init__(
        self,
        model: nn.Module,
        input_channels: int,
        evaluation_audios_dir: str,
        target_source_type: str,
        sample_rate: int,
        mono: bool,
        segment_samples: int,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        logger: pl.loggers.TensorBoardLogger,
        statistics_container: StatisticsContainer,
    ):
        r"""Callback to evaluate every #save_step_frequency steps.

        Args:
            model: nn.Module
            input_channels: int
            evaluation_audios_dir: str, directory containing audios for evaluation
            target_source_type: str, e.g., 'violin'
            sample_rate: int
            mono: bool
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
            evaluate_step_frequency: int, evaluate every #save_step_frequency steps
            logger: pl.loggers.TensorBoardLogger
            statistics_container: StatisticsContainer
        """
        self.model = model
        self.target_source_type = target_source_type
        self.sample_rate = sample_rate
        self.mono = mono
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger
        self.statistics_container = statistics_container

        self.evaluation_audios_dir = evaluation_audios_dir

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _) -> NoReturn:
        r"""Evaluate losses on a few mini-batches. Losses are only used for
        observing training, and are not final F1 metrics.
        """

        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            mixture_audios_dir = os.path.join(self.evaluation_audios_dir, 'mixture')
            clean_audios_dir = os.path.join(
                self.evaluation_audios_dir, self.target_source_type
            )

            audio_names = sorted(os.listdir(mixture_audios_dir))

            error_str = "Directory {} does not contain audios for evaluation!".format(
                self.evaluation_audios_dir
            )
            assert len(audio_names) > 0, error_str

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(audio_names)))

            eval_time = time.time()

            sdrs = []

            for n, audio_name in enumerate(audio_names):

                # Load audio.
                mixture_path = os.path.join(mixture_audios_dir, audio_name)
                clean_path = os.path.join(clean_audios_dir, audio_name)

                mixture, origin_fs = librosa.core.load(
                    mixture_path, sr=self.sample_rate, mono=self.mono
                )

                # Target
                clean, origin_fs = librosa.core.load(
                    clean_path, sr=self.sample_rate, mono=self.mono
                )

                if mixture.ndim == 1:
                    mixture = mixture[None, :]
                # (channels_num, audio_length)

                input_dict = {'waveform': mixture}

                # separate
                sep_wav = self.separator.separate(input_dict)
                # (channels_num, audio_length)

                sdr = calculate_sdr(ref=clean, est=sep_wav)

                print("{} SDR: {:.3f}".format(audio_name, sdr))
                sdrs.append(sdr)

            logging.info("-----------------------------")
            logging.info('Avg SDR: {:.3f}'.format(np.mean(sdrs)))

            logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

            statistics = {"sdr": np.mean(sdrs)}
            self.statistics_container.append(global_step, statistics, 'test')
            self.statistics_container.dump()
