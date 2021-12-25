import logging
import os
import time
from typing import Dict, List, NoReturn

import librosa
import musdb
import museval
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

from bytesep.callbacks.base import SaveCheckpointsCallback
from bytesep.dataset_creation.pack_audios_to_hdf5s.musdb18 import preprocess_audio
from bytesep.separate import Separator
from bytesep.utils import StatisticsContainer, read_yaml


def get_musdb18_callbacks(
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
) -> List[pl.Callback]:
    r"""Get MUSDB18 callbacks of a config yaml.

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
    evaluation_callback = configs['train']['evaluation_callback']
    target_source_types = configs['train']['target_source_types']
    input_channels = configs['train']['input_channels']
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
        split='train',
        model=model,
        target_source_types=target_source_types,
        sample_rate=sample_rate,
        input_channels=input_channels,
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    evaluate_test_callback = EvaluationCallback(
        dataset_dir=evaluation_audios_dir,
        split='test',
        model=model,
        target_source_types=target_source_types,
        sample_rate=sample_rate,
        input_channels=input_channels,
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
    if evaluation_callback == "Musdb18":
        return Musdb18EvaluationCallback

    if evaluation_callback == 'Musdb18Conditional':
        return Musdb18ConditionalEvaluationCallback

    else:
        raise NotImplementedError


class Musdb18EvaluationCallback(pl.Callback):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        model: nn.Module,
        target_source_types: str,
        sample_rate: int,
        input_channels: int,
        segment_samples: int,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        logger: pl.loggers.TensorBoardLogger,
        statistics_container: StatisticsContainer,
    ):
        r"""Callback to evaluate every #save_step_frequency steps.

        Args:
            dataset_dir: str
            model: nn.Module
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
            input_channels: int
            split: 'train' | 'test'
            sample_rate: int
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
    def on_batch_end(self, trainer: pl.Trainer, _) -> NoReturn:
        r"""Evaluate separation SDRs of audio recordings."""
        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            sdr_dict = {}

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(self.mus.tracks)))

            eval_time = time.time()

            for track in self.mus.tracks:

                audio_name = track.name

                # Get waveform of mixture.
                mixture = track.audio.T
                # (channels_num, audio_samples)

                mixture = preprocess_audio(
                    audio=mixture,
                    mono=self.mono,
                    origin_sr=track.rate,
                    sr=self.sample_rate,
                    resample_type=self.resample_type,
                )
                # (channels_num, audio_samples)

                target_dict = {}
                sdr_dict[audio_name] = {}

                # Get waveform of all target source types.
                for j, source_type in enumerate(self.target_source_types):
                    # E.g., ['vocals', 'bass', ...]

                    audio = track.targets[source_type].audio.T

                    audio = preprocess_audio(
                        audio=audio,
                        mono=self.mono,
                        origin_sr=track.rate,
                        sr=self.sample_rate,
                        resample_type=self.resample_type,
                    )
                    # (channels_num, audio_samples)

                    target_dict[source_type] = audio
                    # (channels_num, audio_samples)

                # Separate.
                input_dict = {'waveform': mixture}

                sep_wavs = self.separator.separate(input_dict)
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                # Post process separation results.
                sep_wavs = preprocess_audio(
                    audio=sep_wavs,
                    mono=self.mono,
                    origin_sr=self.sample_rate,
                    sr=track.rate,
                    resample_type=self.resample_type,
                )
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                sep_wavs = librosa.util.fix_length(
                    sep_wavs, size=mixture.shape[1], axis=1
                )
                # sep_wavs: (target_sources_num * channels_num, audio_samples)

                sep_wav_dict = get_separated_wavs_from_simo_output(
                    sep_wavs, self.input_channels, self.target_source_types
                )
                # output_dict: dict, e.g., {
                #     'vocals': (channels_num, audio_samples),
                #     'bass': (channels_num, audio_samples),
                #     ...,
                # }

                # Evaluate for all target source types.
                for source_type in self.target_source_types:
                    # E.g., ['vocals', 'bass', ...]

                    # Calculate SDR using museval, input shape should be: (nsrc, nsampl, nchan).
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

            # Calculate median SDRs of all songs.
            for source_type in self.target_source_types:
                # E.g., ['vocals', 'bass', ...]

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


def get_separated_wavs_from_simo_output(x, input_channels, target_source_types) -> Dict:
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
        dataset_dir: str,
        split: str,
        model: nn.Module,
        target_source_types: str,
        sample_rate: int,
        input_channels: int,
        segment_samples: int,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        logger: pl.loggers.TensorBoardLogger,
        statistics_container: StatisticsContainer,
    ):
        r"""Callback to evaluate every #save_step_frequency steps.

        Args:
            dataset_dir: str
            split: 'train' | 'test'
            model: nn.Module
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
            sample_rate: int
            input_channels: int
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
    def on_batch_end(self, trainer: pl.Trainer, _) -> NoReturn:
        r"""Evaluate separation SDRs of audio recordings."""
        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            sdr_dict = {}

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(self.mus.tracks)))

            eval_time = time.time()

            for track in self.mus.tracks:

                audio_name = track.name

                # Get waveform of mixture.
                mixture = track.audio.T
                # (channels_num, audio_samples)

                mixture = preprocess_audio(
                    audio=mixture,
                    mono=self.mono,
                    origin_sr=track.rate,
                    sr=self.sample_rate,
                    resample_type=self.resample_type,
                )
                # (channels_num, audio_samples)

                target_dict = {}
                sdr_dict[audio_name] = {}

                # Get waveform of all target source types.
                for j, source_type in enumerate(self.target_source_types):
                    # E.g., ['vocals', 'bass', ...]

                    audio = track.targets[source_type].audio.T

                    audio = preprocess_audio(
                        audio=audio,
                        mono=self.mono,
                        origin_sr=track.rate,
                        sr=self.sample_rate,
                        resample_type=self.resample_type,
                    )
                    # (channels_num, audio_samples)

                    target_dict[source_type] = audio
                    # (channels_num, audio_samples)

                    condition = np.zeros(len(self.target_source_types))
                    condition[j] = 1

                    input_dict = {'waveform': mixture, 'condition': condition}

                    sep_wav = self.separator.separate(input_dict)
                    # sep_wav: (channels_num, audio_samples)

                    sep_wav = preprocess_audio(
                        audio=sep_wav,
                        mono=self.mono,
                        origin_sr=self.sample_rate,
                        sr=track.rate,
                        resample_type=self.resample_type,
                    )
                    # sep_wav: (channels_num, audio_samples)

                    sep_wav = librosa.util.fix_length(
                        sep_wav, size=mixture.shape[1], axis=1
                    )
                    # sep_wav: (target_sources_num * channels_num, audio_samples)

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

            # Calculate median SDRs of all songs.
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
