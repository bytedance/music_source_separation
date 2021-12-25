from typing import Dict, List, NoReturn, Optional

import h5py
import librosa
import numpy as np
import torch
from pytorch_lightning.core.datamodule import LightningDataModule

from bytesep.data.augmentors import Augmentor
from bytesep.data.samplers import DistributedSamplerWrapper
from bytesep.utils import int16_to_float32


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_sampler: object,
        train_dataset: object,
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

        # SegmentSampler is used for sampling segment indexes for training.
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
    def __init__(
        self,
        input_source_types: List[str],
        target_source_types: List[str],
        paired_input_target_data: bool,
        input_channels: int,
        augmentor: Augmentor,
        segment_samples: int,
    ):
        r"""Used for getting data according to a meta.

        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        self.input_source_types = input_source_types
        self.paired_input_target_data = paired_input_target_data
        self.input_channels = input_channels
        self.augmentor = augmentor
        self.segment_samples = segment_samples

        if paired_input_target_data:
            self.source_types = list(set(input_source_types) | set(target_source_types))

        else:
            self.source_types = input_source_types

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
        data_dict = {}

        for source_type in self.source_types:
            # E.g., ['vocals', 'accompaniment']

            waveforms = []  # Audio segments to be mix-audio augmented.

            for m in meta[source_type]:
                # E.g., {
                #     'hdf5_path': '.../song_A.h5',
                #     'key_in_hdf5': 'vocals',
                #     'begin_sample': '13406400',
                #     'end_sample': 13538700,
                # }

                hdf5_path = m['hdf5_path']
                key_in_hdf5 = m['key_in_hdf5']
                bgn_sample = m['begin_sample']
                end_sample = m['end_sample']

                with h5py.File(hdf5_path, 'r') as hf:

                    if source_type == 'audioset':
                        index_in_hdf5 = m['index_in_hdf5']
                        waveform = int16_to_float32(
                            hf['waveform'][index_in_hdf5][bgn_sample:end_sample]
                        )
                        waveform = waveform[None, :]
                    else:
                        waveform = int16_to_float32(
                            hf[key_in_hdf5][:, bgn_sample:end_sample]
                        )

                if self.paired_input_target_data:
                    # TODO
                    pass

                else:
                    if self.augmentor:
                        waveform = self.augmentor(waveform, source_type)

                if source_type in self.input_source_types:
                    waveform = self.match_waveform_to_input_channels(
                        waveform=waveform, input_channels=self.input_channels
                    )
                    # (input_channels, segments_num)

                waveform = librosa.util.fix_length(
                    waveform, size=self.segment_samples, axis=1
                )

                waveforms.append(waveform)
            # E.g., waveforms: [(input_channels, audio_samples), (input_channels, audio_samples)]

            # mix-audio augmentation
            data_dict[source_type] = np.sum(waveforms, axis=0)
            # data_dict[source_type]: (input_channels, audio_samples)

        # data_dict looks like: {
        #     'voclas': (input_channels, audio_samples),
        #     'accompaniment': (input_channels, audio_samples)
        # }

        return data_dict

    def match_waveform_to_input_channels(
        self,
        waveform: np.array,
        input_channels: int,
    ) -> np.array:
        r"""Match waveform to channels num.

        Args:
            waveform: (input_channels, segments_num)
            input_channels: int

        Outputs:
            output: (new_input_channels, segments_num)
        """
        waveform_channels = waveform.shape[0]

        if waveform_channels == input_channels:
            return waveform

        elif waveform_channels < input_channels:
            assert waveform_channels == 1
            return np.tile(waveform, (input_channels, 1))

        else:
            assert input_channels == 1
            return np.mean(waveform, axis=0)[None, :]


def collate_fn(list_data_dict: List[Dict]) -> Dict:
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {'vocals': (input_channels, segment_samples),
             'accompaniment': (input_channels, segment_samples),
             'mixture': (input_channels, segment_samples)
            },
            {'vocals': (input_channels, segment_samples),
             'accompaniment': (input_channels, segment_samples),
             'mixture': (input_channels, segment_samples)
            },
            ...]

    Returns:
        data_dict: e.g. {
            'vocals': (batch_size, input_channels, segment_samples),
            'accompaniment': (batch_size, input_channels, segment_samples),
            'mixture': (batch_size, input_channels, segment_samples)
            }
    """
    data_dict = {}

    for key in list_data_dict[0].keys():
        data_dict[key] = torch.Tensor(
            np.array([data_dict[key] for data_dict in list_data_dict])
        )

    return data_dict
