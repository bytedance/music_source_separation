import h5py
import logging
import os
from typing import List
import pathlib
import time
import librosa
from pesq import pesq
import pysepm
import glob

import museval
import pandas as pd
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

from music_source_separation.callbacks.base_callbacks import SaveCheckpointsCallback
from music_source_separation.utils import (
    read_yaml,
    StatisticsContainer,
    int16_to_float32,
)
from music_source_separation.inference import Separator


def get_voicebank_demand_callbacks(
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
):

    dataset_dir = "/home/tiger/datasets/voicebank-demand"

    configs = read_yaml(config_yaml)
    # target_source_type = configs['train']['target_source_types'][0]
    target_source_types = configs['train']['target_source_types']
    assert len(target_source_types) == 1
    target_source_type = target_source_types[0]
    assert target_source_type == 'speech'
    input_channels = configs['train']['channels']
    # test_hdf5s_dir = os.path.join(workspace, configs['evaluate']['test'])
    clean_dir = os.path.join(dataset_dir, configs['evaluate']['test']['clean_dir'])
    noisy_dir = os.path.join(dataset_dir, configs['evaluate']['test']['noisy_dir'])
    test_segment_seconds = configs['evaluate']['segment_seconds']
    sample_rate = configs['train']['sample_rate']
    test_segment_samples = int(test_segment_seconds * sample_rate)
    test_batch_size = configs['evaluate']['batch_size']

    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']

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
        # input_channels=input_channels,
        # hdf5s_dir=test_hdf5s_dir,
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        # split='test',
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
        # hdf5s_dir: str,
        input_channels,
        clean_dir,
        noisy_dir,
        sample_rate,
        segment_samples: int,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        logger,
        statistics_container: StatisticsContainer,
    ):
        r"""Callback to evaluate every #save_step_frequency steps.

        Args:
            model: nn.Module
            target_source_type: str, e.g., 'vocals'
            hdf5s_dir, str, directory containing hdf5 files for evaluation.
            split: 'train' | 'test'
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
            evaluate_step_frequency: int, evaluate every #save_step_frequency steps
            logger: object
            statistics_container: StatisticsContainer
        """
        self.model = model
        # self.mono = True if input_channels == 1 else False
        self.mono = True
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger
        self.statistics_container = statistics_container

        # self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir

        self.EVALUATION_SAMPLE_RATE = 16000

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _):
        r"""Evaluate losses on a few mini-batches. Losses are only used for
        observing training, and are not final F1 metrics.
        """

        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            # audio_names = sorted(os.listdir(self.clean_dir))
            audio_names = sorted(glob.glob('{}/*.wav'.format(self.clean_dir)))
            pesqs, csigs, cbaks, covls, ssnrs = [], [], [], [], []

            for n, audio_name in enumerate(audio_names):

                # Load audio.
                clean_path = os.path.join(self.clean_dir, audio_name)
                mixture_path = os.path.join(self.noisy_dir, audio_name)

                mixture, _ = librosa.core.load(
                    mixture_path, sr=self.sample_rate, mono=self.mono
                )

                if mixture.ndim == 1:
                    mixture = mixture[None, :]
                # (channels, audio_length)

                # separate
                sep_wav = self.separator.separate(mixture)
                # (channels, audio_length)

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
                print(n, audio_name, pesq_, csig, cbak, covl, ssnr)

                if n == 10:
                    break

            print('Avg PESQ: {:.3f}'.format(np.mean(pesqs)))
            print('Avg CSIG: {:.3f}'.format(np.mean(csigs)))
            print('Avg CBAK: {:.3f}'.format(np.mean(cbaks)))
            print('Avg COVL: {:.3f}'.format(np.mean(covls)))
            print('Avg SSNR: {:.3f}'.format(np.mean(ssnrs)))

            sdr_dict = {}

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(audio_names)))

            # self.logger.experiment.add_scalar(
            #     "SDR/{}".format(self.split), median_sdr, global_step
            # )

            statistics = {"pesq": np.mean(pesqs)}
            self.statistics_container.append(global_step, statistics, 'test')
            self.statistics_container.dump()
