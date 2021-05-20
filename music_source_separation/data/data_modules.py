from typing import Dict, Optional

import numpy as np
import h5py
import torch
from pytorch_lightning.core.datamodule import LightningDataModule

from music_source_separation.data.samplers import DistributedSamplerWrapper
from music_source_separation.data.augmentors import Augmentor
from music_source_separation.utils import int16_to_float32


class DataModule(LightningDataModule):
    def __init__(
        self,
        indexes_path: str,
        max_random_shift: int,
        mixaudio_dict: Dict,
        augmentor: Augmentor,
        Sampler,
        batch_size: int,
        steps_per_epoch: int,
        num_workers: int,
        distributed: bool,
    ):
        r"""Data module.

        Args:
            indexes_path: str, path of indexes dict
            mixaudio_dict, dict, including hyper-parameters for mix-audio data
                augmentation, e.g., {'voclas': 2, 'accompaniment': 2}
            augmentor: Augmentor
            Sampler: Sampler class
            batch_size, int, e.g., 12
            steps_per_epoch: int, #steps_per_epoch is called an `epoch`
                e.g., 10000
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self.indexes_path = indexes_path
        self.max_random_shift = max_random_shift
        self.mixaudio_dict = mixaudio_dict
        self.Sampler = Sampler
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.distributed = distributed

        self.train_dataset = Dataset(augmentor)

    def setup(self, stage: Optional[str] = None):
        r"""called on every device."""

        # SegmentSampler is used for selecting segments for training.
        # On multipythole devices, each SegmentSampler samples a part of mini-batch
        # data.
        _train_sampler = self.Sampler(
            indexes_path=self.indexes_path,
            max_random_shift=self.max_random_shift,
            mixaudio_dict=self.mixaudio_dict,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
        )

        if self.distributed:
            self.train_sampler = DistributedSamplerWrapper(_train_sampler)
        else:
            self.train_sampler = _train_sampler

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader


class Dataset:
    def __init__(self, augmentor):
        r"""Used for returning data according to a meta."""
        self.augmentor = augmentor

    def __getitem__(self, meta):
        r"""Return data according to a meta. E.g., a meta looks like:

            {'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
             'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}.

        Then, vocals segments of song_A and song_B will be mixed (mix-audio augmentation),
        and accompaniment segments of song_C and song_B will be mixed (mix-audio augmentation).
        Finally, mixture is created by summing vocals and accompaniment.

        Args:
            meta: dict, e.g.,
                {'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
                 'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}

        Returns:
            data_dict: dict, e.g.,
        """

        source_types = meta.keys()
        data_dict = {}

        for source_type in source_types:

            waveforms = []  # Audio segments to be mix-audio augmented.

            for m in meta[source_type]:
                [hdf5_path, start_sample, end_sample] = m

                with h5py.File(hdf5_path, 'r') as hf:

                    waveform = int16_to_float32(hf[source_type][:, start_sample:end_sample])

                    waveform = self.augmentor(waveform)

                waveforms.append(waveform)
            # E.g., waveforms: [(channels_num, audio_samples), (channels_num, audio_samples)]

            # mix-audio augmentation
            data_dict[source_type] = np.sum(waveforms, axis=0)
            # data_dict[source_type]: (channels_num, audio_samples)

        # data_dict looks like: {
        #     'voclas': (channels_num, audio_samples),
        #     'accompaniment': (channels_num, audio_samples)
        # }

        # Mix segments from different sources.
        mixture = np.sum([data_dict[source_type] for source_type in source_types], axis=0)
        data_dict['mixture'] = mixture
        # shape: (channels_num, audio_samples)

        return data_dict


def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {'vocals': (channels_num, segment_samples),
             'accompaniment': (channels_num, segment_samples),
             'mixture': (channels_num, segment_samples)
            },
            {'vocals': (channels_num, segment_samples),
             'accompaniment': (channels_num, segment_samples),
             'mixture': (channels_num, segment_samples)
            },
            ...]

    Returns:
        data_dict: e.g. {
            'vocals': (batch_size, channels_num, segment_samples),
            'accompaniment': (batch_size, channels_num, segment_samples),
            'mixture': (batch_size, channels_num, segment_samples)
            }
    """

    data_dict = {}
    for key in list_data_dict[0].keys():
        data_dict[key] = torch.Tensor(np.array([data_dict[key] for data_dict in list_data_dict]))

    return data_dict
