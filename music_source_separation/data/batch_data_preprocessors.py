from typing import Dict, List

import torch


class BasicBatchDataPreprocessor:
    def __init__(self, target_source_types: List[str]):
        r"""Batch data preprocessor. Used for prepare mixtures and targets for
        training. If there are multiple target source types, then the waveforms
        of those sources are stacked along the channel dimension.

        Args:
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
        """
        self.target_source_types = target_source_types

    def __call__(self, batch_data_dict: Dict) -> List[Dict]:
        r"""Format waveforms and targets for training.

        Args:
            batch_data_dict: dict, e.g., {
                'mixture': (batch_size, channels_num, segment_samples),
                'vocals': (batch_size, channels_num, segment_samples),
                'bass': (batch_size, channels_num, segment_samples),
                ...,
            }

        Returns:
            input_dict: dict, e.g., {
                'waveform': (batch_size, channels_num, segment_samples),
            }
            output_dict: dict, e.g., {
                'target': (batch_size, target_sources_num * channels_num, segment_samples)
            }
        """
        mixtures = batch_data_dict['mixture']
        # mixtures: (batch_size, channels_num, segment_samples)

        # Concatenate waveforms of multiple targets along the channel axis.
        targets = torch.cat(
            [batch_data_dict[source_type] for source_type in self.target_source_types],
            dim=1,
        )
        # targets: (batch_size, target_sources_num * channels_num, segment_samples)

        input_dict = {'waveform': mixtures}

        target_dict = {'waveform': targets}

        return input_dict, target_dict


class ConditionalSisoBatchDataPreprocessor:
    def __init__(self, target_source_types: List[str]):
        r"""Conditional single input single output (SISO) batch data
        preprocessor. Used for prepare mixtures and targets for training. The
        input contains both waveforms and conditions.

        Args:
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
        """
        self.target_source_types = target_source_types

    def __call__(self, batch_data_dict: Dict) -> List[Dict]:
        r"""Format waveforms and targets for training.

        Args:
            batch_data_dict: dict, e.g., {
                'mixture': (batch_size, channels_num, segment_samples),
                'vocals': (batch_size, channels_num, segment_samples),
                'bass': (batch_size, channels_num, segment_samples),
                ...,
            }

        Returns:
            input_dict: dict, e.g., {
                'waveform': (batch_size, channels_num, segment_samples),
                'condition': (batch_size, target_sources_num),
            }
            output_dict: dict, e.g., {
                'target': (batch_size, channels_num, segment_samples)
            }
        """

        batch_size = len(batch_data_dict['mixture'])
        target_sources_num = len(self.target_source_types)

        assert (
            batch_size % target_sources_num == 0
        ), "Batch size should be \
            evenly divided by target sources number."

        mixtures = batch_data_dict['mixture']
        # mixtures: (batch_size, channels_num, segment_samples)

        conditions = torch.zeros(batch_size, target_sources_num).to(mixtures.device)
        # conditions: (batch_size, target_sources_num)

        targets = []

        for n in range(batch_size):

            k = n % target_sources_num  # source class index
            source_type = self.target_source_types[k]

            targets.append(batch_data_dict[source_type][n])

            conditions[n, k] = 1

        # conditions will looks like:
        # [[1, 0, 0, 0],
        #  [0, 1, 0, 0],
        #  [0, 0, 1, 0],
        #  [0, 0, 0, 1],
        #  [1, 0, 0, 0],
        #  [0, 1, 0, 0],
        #  ...,
        # ]

        targets = torch.stack(targets, dim=0)
        # targets: (batch_size, channels_num, segment_samples)

        input_dict = {
            'waveform': mixtures,
            'condition': conditions,
        }

        target_dict = {'target': targets}

        return input_dict, target_dict


def get_batch_data_preprocessor_class(batch_data_preprocessor_type: str) -> object:
    r"""Get batch data preprocessor class."""
    if batch_data_preprocessor_type == 'BasicBatchDataPreprocessor':
        return BasicBatchDataPreprocessor

    elif batch_data_preprocessor_type == 'ConditionalSisoBatchDataPreprocessor':
        return ConditionalSisoBatchDataPreprocessor

    else:
        raise NotImplementedError
