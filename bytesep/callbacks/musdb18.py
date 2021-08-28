import logging
import os
import pathlib
import time
from typing import List

import librosa
import musdb
import h5py
import museval
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

from bytesep.dataset_creation.pack_audios_to_hdf5s.musdb18 import preprocess_audio
from bytesep.callbacks.base_callbacks import SaveCheckpointsCallback
from bytesep.inference import Separator
from bytesep.utils import (
    StatisticsContainer,
    int16_to_float32,
    read_yaml,
)


def get_musdb18_callbacks(
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
) -> List[pl.Callback]:
    """Get MUSDB18 callbacks of a config yaml.

    Args:
        config_yaml: str
        workspace: str
        checkpoints_dir: str
        statistics_dir: str
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    configs = read_yaml(config_yaml)
    task_name = configs['task_name']
    evaluation_callback = configs['train']['evaluation_callback']
    target_source_types = configs['train']['target_source_types']
    input_channels = configs['train']['channels']
    evaluation_audios_dir = os.path.join(workspace, "evaluation_audios", task_name)
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

    # evaluation callback
    EvaluationCallback = _get_evaluation_callback_class(evaluation_callback)

    # statistics container
    statistics_container = StatisticsContainer(statistics_path)

    # evaluation callback
    evaluate_train_callback = EvaluationCallback(
        dataset_dir=evaluation_audios_dir,
        model=model,
        target_source_types=target_source_types,
        input_channels=input_channels,
        sample_rate=sample_rate,
        split='train',
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    evaluate_test_callback = EvaluationCallback(
        dataset_dir=evaluation_audios_dir,
        model=model,
        target_source_types=target_source_types,
        input_channels=input_channels,
        sample_rate=sample_rate,
        split='test',
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    # callbacks = [save_checkpoints_callback, evaluate_train_callback, evaluate_test_callback]
    callbacks = [save_checkpoints_callback, evaluate_test_callback]

    return callbacks


def _get_evaluation_callback_class(evaluation_callback) -> pl.Callback:
    r"""Get evaluation callback class."""
    if evaluation_callback == "Musdb18EvaluationCallback":
        return Musdb18EvaluationCallback

    if evaluation_callback == 'Musdb18ConditionalEvaluationCallback':
        return Musdb18ConditionalEvaluationCallback

    else:
        raise NotImplementedError


class Musdb18EvaluationCallback(pl.Callback):
    def __init__(
        self,
        dataset_dir: str,
        model: nn.Module,
        target_source_types: str,
        input_channels,
        split: str,
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
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
            input_channels: int
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
        self.target_source_types = target_source_types
        self.input_channels = input_channels
        self.sample_rate = sample_rate
        self.split = split
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger
        self.statistics_container = statistics_container
        self.mono = input_channels == 1
        self.resample_type = "kaiser_fast"

        self.mus = musdb.DB(root=dataset_dir, subsets=[split])

        error_msg = "The directory {} is empty!".format(dataset_dir)
        assert len(self.mus) > 0, error_msg

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _) -> None:
        r"""Evaluate separation SDRs of audio recordings."""
        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            sdr_dict = {}

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(self.mus.tracks)))

            eval_time = time.time()

            for track in self.mus.tracks:

                audio_name = track.name

                mixture = track.audio.T
                # (channels_num, audio_samples)

                mixture = preprocess_audio(
                    audio=mixture, 
                    mono=self.mono, 
                    origin_sr=track.rate, 
                    sr=self.sample_rate, 
                    resample_type=self.resample_type
                )
                # (channels_num, audio_samples)

                target_dict = {}
                sdr_dict[audio_name] = {}

                for j, source_type in enumerate(self.target_source_types):

                    audio = track.targets[source_type].audio.T

                    audio = preprocess_audio(
                        audio=audio, 
                        mono=self.mono, 
                        origin_sr=track.rate, 
                        sr=self.sample_rate, 
                        resample_type=self.resample_type
                    )
                    # (channels_num, audio_samples)

                    target_dict[source_type] = audio
                    # (channels_num, audio_samples)

                input_dict = {'waveform': mixture}

                sep_wavs = self.separator.separate(input_dict)
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                sep_wavs = preprocess_audio(
                    audio=sep_wavs, 
                    mono=self.mono, 
                    origin_sr=self.sample_rate, 
                    sr=track.rate, 
                    resample_type=self.resample_type
                )
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                sep_wavs = librosa.util.fix_length(sep_wavs, size=mixture.shape[1], axis=1)
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                sep_wav_dict = get_separated_wavs_from_simo_output(
                    sep_wavs, self.input_channels, self.target_source_types
                )
                # output_dict: dict, e.g., {
                #     'vocals': (channels_num, audio_samples),
                #     'bass': (channels_num, audio_samples),
                #     ...,
                # }

                for source_type in self.target_source_types:
                    # Calculate SDR using museval, input shape should be: (nsrc, nsampl, nchan)

                    (sdrs, _, _, _) = museval.evaluate(
                        [target_dict[source_type].T], [sep_wav_dict[source_type].T]
                    )

                    sdr = np.nanmedian(sdrs)
                    sdr_dict[audio_name][source_type] = sdr

                    logging.info(
                        "{}, {}, sdr: {:.3f}".format(audio_name, source_type, sdr)
                    )

            logging.info("-----------------------------")
            median_sdr_dict = {}

            for source_type in self.target_source_types:

                median_sdr = np.median(
                    [
                        sdr_dict[audio_name][source_type]
                        for audio_name in sdr_dict.keys()
                    ]
                )

                median_sdr_dict[source_type] = median_sdr

                logging.info(
                    "Step: {}, {}, Median SDR: {:.3f}".format(
                        global_step, source_type, median_sdr
                    )
                )

            logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

            statistics = {"sdr_dict": sdr_dict, "median_sdr_dict": median_sdr_dict}
            self.statistics_container.append(global_step, statistics, self.split)
            self.statistics_container.dump()


def get_separated_wavs_from_simo_output(x, input_channels, target_source_types):
    r"""Get separated waveforms of target sources from a single input multiple
    output (SIMO) system.

    Args:
        x: (target_sources_num * channels_num, audio_samples)
        input_channels: int
        target_source_types: List[str], e.g., ['vocals', 'bass', ...]

    Returns:
        output_dict: dict, e.g., {
            'vocals': (channels_num, audio_samples),
            'bass': (channels_num, audio_samples),
            ...,
        }
    """
    output_dict = {}

    for j, source_type in enumerate(target_source_types):
        output_dict[source_type] = x[j * input_channels : (j + 1) * input_channels]

    return output_dict


class Musdb18ConditionalEvaluationCallback(pl.Callback):
    def __init__(
        self,
        model: nn.Module,
        target_source_types: str,
        input_channels,
        hdf5s_dir: str,
        split: str,
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
            target_source_types: str, e.g., ['vocals', 'bass', ...]
            input_channels: int
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
        self.target_source_types = target_source_types
        self.input_channels = input_channels
        self.hdf5s_dir = hdf5s_dir
        self.split = split
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger
        self.statistics_container = statistics_container

        self.hdf5_names = sorted(os.listdir(self.hdf5s_dir))

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _) -> None:
        r"""Evaluate separation SDRs of audio recordings."""

        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            sdr_dict = {}

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(self.hdf5_names)))

            eval_time = time.time()

            for hdf5_name in self.hdf5_names:

                hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
                audio_name = pathlib.Path(hdf5_name).stem
                sdr_dict[audio_name] = {}

                target_dict = {}

                with h5py.File(hdf5_path, "r") as hf:

                    mixture = int16_to_float32(hf["mixture"][:])
                    # mixture: (channels_num, audio_samples)

                    for j, source_type in enumerate(self.target_source_types):
                        # E.g., ['vocals', 'bass', ...]

                        target_dict[source_type] = int16_to_float32(hf[source_type][:])

                        condition = np.zeros(len(self.target_source_types))
                        condition[j] = 1
                        input_dict = {'waveform': mixture, 'condition': condition}

                        sep_wav = self.separator.separate(input_dict)
                        # sep_wav: (channels_num, audio_samples)

                        # Calculate SDR using museval, input shape should be: (nsrc, nsampl, nchan)
                        (sdrs, _, _, _) = museval.evaluate(
                            [target_dict[source_type].T], [sep_wav.T]
                        )

                        sdr = np.nanmedian(sdrs)
                        sdr_dict[audio_name][source_type] = sdr

                        logging.info(
                            "{}, {}, sdr: {:.3f}".format(audio_name, source_type, sdr)
                        )

            logging.info("-----------------------------")
            median_sdr_dict = {}

            for source_type in self.target_source_types:
                median_sdr = np.median(
                    [
                        sdr_dict[audio_name][source_type]
                        for audio_name in sdr_dict.keys()
                    ]
                )
                median_sdr_dict[source_type] = median_sdr

                logging.info(
                    "Step: {}, {}, Median SDR: {:.3f}".format(
                        global_step, source_type, median_sdr
                    )
                )

            logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

            statistics = {"sdr_dict": sdr_dict, "median_sdr_dict": median_sdr_dict}
            self.statistics_container.append(global_step, statistics, self.split)
            self.statistics_container.dump()
