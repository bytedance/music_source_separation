import pathlib
import numpy as np
import museval
import time
import logging
import os
import h5py
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from music_source_separation.callbacks.base_callbacks import SaveCheckpointsCallback
from music_source_separation.utils import (
    read_yaml,
    StatisticsContainer,
    int16_to_float32,
)
from music_source_separation.inference import Separator


def get_musdb18_callbacks(
    config_yaml: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
):

    configs = read_yaml(config_yaml)
    # target_source_type = configs['train']['target_source_types'][0]
    evaluation_callback = configs['train']['evaluation_callback']
    target_source_types = configs['train']['target_source_types']
    input_channels = configs['train']['channels']
    test_hdf5s_dir = os.path.join(workspace, configs['evaluate']['test'])
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

    EvaluationCallback = _get_evaluation_callback_class(evaluation_callback)

    # evaluation callback
    evaluate_test_callback = EvaluationCallback(
        model=model,
        target_source_types=target_source_types,
        input_channels=input_channels,
        hdf5s_dir=test_hdf5s_dir,
        split='test',
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    callbacks = [save_checkpoints_callback, evaluate_test_callback]

    return callbacks


def _get_evaluation_callback_class(evaluation_callback):

    if evaluation_callback == "Musdb18EvaluationCallback":
        return Musdb18EvaluationCallback

    if evaluation_callback == 'Musdb18ConditionalEvaluationCallback':
        return Musdb18ConditionalEvaluationCallback

    else:
        raise NotImplementedError


class Musdb18EvaluationCallback(pl.Callback):
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

                    for j, source_type in enumerate(self.target_source_types):
                        target_dict[source_type] = int16_to_float32(hf[source_type][:])

                input_dict = {'waveform': mixture}

                sep_wavs = self.separator.separate(input_dict)

                sep_wav_dict = add(
                    sep_wavs, self.input_channels, self.target_source_types
                )

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


def add(x, input_channels, target_source_types):

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

                    for j, source_type in enumerate(self.target_source_types):
                        target_dict[source_type] = int16_to_float32(hf[source_type][:])

                        condition = np.zeros(len(self.target_source_types))
                        condition[j] = 1
                        input_dict = {'waveform': mixture, 'condition': condition}

                        sep_wav = self.separator.separate(input_dict)

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
