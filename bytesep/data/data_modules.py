from typing import Dict, List, Optional, NoReturn

import h5py
import librosa
import numpy as np
import torch
from pytorch_lightning.core.datamodule import LightningDataModule

from bytesep.data.samplers import DistributedSamplerWrapper
from bytesep.utils import int16_to_float32


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_sampler,
        train_dataset,
        num_workers: int,
        distributed: bool,
    ):
        r"""Data module.

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self._train_sampler = train_sampler
        self.train_dataset = train_dataset
        self.num_workers = num_workers
        self.distributed = distributed

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # SegmentSampler is used for selecting segments for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data.
        if self.distributed:
            self.train_sampler = DistributedSamplerWrapper(self._train_sampler)

        else:
            self.train_sampler = self._train_sampler

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_loader


class Dataset:
    def __init__(self, augmentor, segment_samples: int):
        r"""Used for getting data according to a meta.

        Args:
            augmentor: Augmentor class
            segment_samples: int
        """
        self.augmentor = augmentor
        self.segment_samples = segment_samples

    def __getitem__(self, meta: Dict) -> Dict:
        r"""Return data according to a meta. E.g., an input meta looks like: {
            'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
            'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}.
        }

        Then, vocals segments of song_A and song_B will be mixed (mix-audio augmentation).
        Accompaniment segments of song_C and song_B will be mixed (mix-audio augmentation).
        Finally, mixture is created by summing vocals and accompaniment.

        Args:
            meta: dict, e.g., {
                'vocals': [['song_A.h5', 6332760, 6465060], ['song_B.h5', 198450, 330750]],
                'accompaniment': [['song_C.h5', 24232920, 24365250], ['song_D.h5', 1569960, 1702260]]}
            }

        Returns:
            data_dict: dict, e.g., {
                'vocals': (channels, segments_num),
                'accompaniment': (channels, segments_num),
                'mixture': (channels, segments_num),
            }
        """
        source_types = meta.keys()
        data_dict = {}

        for source_type in source_types:
            # E.g., ['vocals', 'bass', ...]

            waveforms = []  # Audio segments to be mix-audio augmented.

            for m in meta[source_type]:
                # E.g., ['.../song_A.h5', 3995460, 4127760]

                [hdf5_path, key, start_sample, end_sample] = m

                with h5py.File(hdf5_path, 'r') as hf:

                    waveform = int16_to_float32(
                        hf[key][:, start_sample:end_sample]
                    )

                    if self.augmentor:
                        waveform = self.augmentor(waveform)

                    waveform = librosa.util.fix_length(
                        waveform, size=self.segment_samples, axis=1
                    )
                    # (channels_num, segments_num)

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
        mixture = np.sum(
            [data_dict[source_type] for source_type in source_types], axis=0
        )
        data_dict['mixture'] = mixture
        # shape: (channels_num, audio_samples)

        return data_dict


def collate_fn(list_data_dict: List[Dict]) -> Dict:
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
        data_dict[key] = torch.Tensor(
            np.array([data_dict[key] for data_dict in list_data_dict])
        )

    return data_dict
