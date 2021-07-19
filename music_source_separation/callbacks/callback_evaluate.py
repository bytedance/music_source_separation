import h5py
import logging
import os
import pathlib
import time

import museval
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

from music_source_separation.utils import int16_to_float32, StatisticsContainer
from music_source_separation.inference import Separator


class CallbackEvaluation(pl.Callback):
    def __init__(
        self,
        model: nn.Module,
        target_source_type: str,
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
        self.target_source_type = target_source_type
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
    def on_batch_end(self, trainer: pl.Trainer, _):
        r"""Evaluate separation SDRs of audio recordings."""

        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            sdr_dict = {}

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(self.hdf5_names)))

            eval_time = time.time()

            for hdf5_name in self.hdf5_names:
                hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)

                with h5py.File(hdf5_path, "r") as hf:
                    mixture = int16_to_float32(hf["mixture"][:])
                    target = int16_to_float32(hf[self.target_source_type][:])

                sep_wav = self.separator.separate(mixture)

                # Calculate SDR using museval, input shape should be: (nsrc, nsampl, nchan)
                (sdrs, _, _, _) = museval.evaluate([target.T], [sep_wav.T])

                sdr = np.nanmedian(sdrs)

                audio_name = pathlib.Path(hdf5_name).stem
                sdr_dict[audio_name] = sdr
                logging.info("{}, sdr: {:.3f}".format(audio_name, sdr))

            median_sdr = np.median(
                [sdr_dict[audio_name] for audio_name in sdr_dict.keys()]
            )

            print()
            logging.info("Step: {}, Median SDR: {:.3f}".format(global_step, median_sdr))
            logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

            self.logger.experiment.add_scalar(
                "SDR/{}".format(self.split), median_sdr, global_step
            )

            statistics = {"sdr_dict": sdr_dict, "median_sdr": median_sdr}
            self.statistics_container.append(global_step, statistics, self.split)
            self.statistics_container.dump()
