import logging
import os
import time
from typing import List, NoReturn

import librosa
import numpy as np
import pysepm
import pytorch_lightning as pl
import torch.nn as nn
from pesq import pesq
from pytorch_lightning.utilities import rank_zero_only

from bytesep.callbacks.base_callbacks import SaveCheckpointsCallback
from bytesep.inference import Separator
from bytesep.utils import StatisticsContainer, read_yaml


def get_voicebank_demand_callbacks(
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
    evaluation_audios_dir = os.path.join(workspace, "evaluation_audios", task_name)
    sample_rate = configs['train']['sample_rate']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    test_batch_size = configs['evaluate']['batch_size']
    test_segment_seconds = configs['evaluate']['segment_seconds']

    test_segment_samples = int(test_segment_seconds * sample_rate)
    assert len(target_source_types) == 1
    target_source_type = target_source_types[0]
    assert target_source_type == 'speech'

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
        input_channels=input_channels,
        sample_rate=sample_rate,
        evaluation_audios_dir=evaluation_audios_dir,
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    callbacks = [save_checkpoints_callback, evaluate_test_callback]

    return callbacks


class EvaluationCallback(pl.Callback):
    def __init__(
        self,
        model: nn.Module,
        input_channels: int,
        evaluation_audios_dir,
        sample_rate: int,
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
            sample_rate: int
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
            evaluate_step_frequency: int, evaluate every #save_step_frequency steps
            logger: pl.loggers.TensorBoardLogger
            statistics_container: StatisticsContainer
        """
        self.model = model
        self.mono = True
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger
        self.statistics_container = statistics_container

        self.clean_dir = os.path.join(evaluation_audios_dir, "clean_testset_wav")
        self.noisy_dir = os.path.join(evaluation_audios_dir, "noisy_testset_wav")

        self.EVALUATION_SAMPLE_RATE = 16000  # Evaluation sample rate of the
        # Voicebank-Demand task.

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _) -> NoReturn:
        r"""Evaluate losses on a few mini-batches. Losses are only used for
        observing training, and are not final F1 metrics.
        """

        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            audio_names = sorted(
                [
                    audio_name
                    for audio_name in sorted(os.listdir(self.clean_dir))
                    if audio_name.endswith('.wav')
                ]
            )

            error_str = "Directory {} does not contain audios for evaluation!".format(
                self.clean_dir
            )
            assert len(audio_names) > 0, error_str

            pesqs, csigs, cbaks, covls, ssnrs = [], [], [], [], []

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(audio_names)))

            eval_time = time.time()

            for n, audio_name in enumerate(audio_names):

                # Load audio.
                clean_path = os.path.join(self.clean_dir, audio_name)
                mixture_path = os.path.join(self.noisy_dir, audio_name)

                mixture, _ = librosa.core.load(
                    mixture_path, sr=self.sample_rate, mono=self.mono
                )

                if mixture.ndim == 1:
                    mixture = mixture[None, :]
                # (channels_num, audio_length)

                # Separate.
                input_dict = {'waveform': mixture}

                sep_wav = self.separator.separate(input_dict)
                # (channels_num, audio_length)

                # Target
                clean, _ = librosa.core.load(
                    clean_path, sr=self.EVALUATION_SAMPLE_RATE, mono=self.mono
                )

                # to mono
                sep_wav = np.squeeze(sep_wav)

                # Resample for evaluation.
                sep_wav = librosa.resample(
                    sep_wav,
                    orig_sr=self.sample_rate,
                    target_sr=self.EVALUATION_SAMPLE_RATE,
                )

                sep_wav = librosa.util.fix_length(sep_wav, size=len(clean), axis=0)
                # (channels, audio_length)

                # Evaluate metrics
                pesq_ = pesq(self.EVALUATION_SAMPLE_RATE, clean, sep_wav, 'wb')

                (csig, cbak, covl) = pysepm.composite(
                    clean, sep_wav, self.EVALUATION_SAMPLE_RATE
                )

                ssnr = pysepm.SNRseg(clean, sep_wav, self.EVALUATION_SAMPLE_RATE)

                pesqs.append(pesq_)
                csigs.append(csig)
                cbaks.append(cbak)
                covls.append(covl)
                ssnrs.append(ssnr)
                print(
                    '{}, {}, PESQ: {:.3f}, CSIG: {:.3f}, CBAK: {:.3f}, COVL: {:.3f}, SSNR: {:.3f}'.format(
                        n, audio_name, pesq_, csig, cbak, covl, ssnr
                    )
                )

            logging.info("-----------------------------")
            logging.info('Avg PESQ: {:.3f}'.format(np.mean(pesqs)))
            logging.info('Avg CSIG: {:.3f}'.format(np.mean(csigs)))
            logging.info('Avg CBAK: {:.3f}'.format(np.mean(cbaks)))
            logging.info('Avg COVL: {:.3f}'.format(np.mean(covls)))
            logging.info('Avg SSNR: {:.3f}'.format(np.mean(ssnrs)))

            logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

            statistics = {"pesq": np.mean(pesqs)}
            self.statistics_container.append(global_step, statistics, 'test')
            self.statistics_container.dump()
