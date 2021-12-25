from typing import Dict, List

import torch
import torch.nn as nn


class MixtureTargetBatchDataPreprocessor(nn.Module):
    def __init__(self, input_source_types: List[str], target_source_types: List[str]):
        r"""Batch data preprocessor. Used for preparing mixtures and targets for
        training. If there are multiple target source types, the waveforms of
        those sources will be stacked along the channel dimension.

        Args:
            input_source_types: List[str], e.g., ['vocals', 'bass', ...]
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
        """
        super(MixtureTargetBatchDataPreprocessor, self).__init__()

        self.input_source_types = input_source_types
        self.target_source_types = target_source_types

    def __call__(self, batch_data_dict: Dict) -> List[Dict]:
        r"""Format waveforms and targets for training.

        Args:
            batch_data_dict: dict, e.g., {
                'mixture': (batch_size, input_channels, segment_samples),
                'vocals': (batch_size, input_channels, segment_samples),
                'bass': (batch_size, input_channels, segment_samples),
                ...,
            }

        Returns:
            input_dict: dict, e.g., {
                'waveform': (batch_size, input_channels, segment_samples),
            }
            output_dict: dict, e.g., {
                'waveform': (batch_size, target_sources_num * output_channels, segment_samples)
            }
        """
        # Get mixture. Sum waveforms all sources.
        stacked_sources = torch.stack(
            [batch_data_dict[source_type] for source_type in self.input_source_types],
            dim=1,
        )
        # input_waveforms: (batch_size, input_sources, input_channels, segment_samples)

        input_waveforms = torch.sum(stacked_sources, dim=1)
        # input_waveforms: (batch_size, input_channels, segment_samples)

        # Concatenate waveforms of multiple targets along the channel axis.
        target_waveforms = torch.cat(
            [batch_data_dict[source_type] for source_type in self.target_source_types],
            dim=1,
        )
        # target_waveform: (batch_size, target_sources_num * output_channels, segment_samples)

        input_dict = {'waveform': input_waveforms}
        target_dict = {'waveform': target_waveforms}

        return input_dict, target_dict


class MixtureTargetConditionalBatchDataPreprocessor:
    def __init__(self, input_source_types: List[str], target_source_types: List[str]):
        r"""Conditional single input single output (SISO) batch data
        preprocessor. Select one target source from several target sources as
        training target and prepare the corresponding conditional vector.

        Args:
            input_source_types: List[str], e.g., ['vocals', 'bass', ...]
            target_source_types: List[str], e.g., ['vocals', 'bass', ...]
        """
        self.input_source_types = input_source_types
        self.target_source_types = target_source_types

        self.target_sources_num = len(self.target_source_types)

    def __call__(self, batch_data_dict: Dict) -> List[Dict]:
        r"""Format waveforms and targets for training.

        Args:
            batch_data_dict: dict, e.g., {
                'mixture': (batch_size, input_channels, segment_samples),
                'vocals': (batch_size, input_channels, segment_samples),
                'bass': (batch_size, input_channels, segment_samples),
                ...,
            }

        Returns:
            input_dict: dict, e.g., {
                'waveform': (batch_size, input_channels, segment_samples),
                'condition': (batch_size, target_sources_num),
            }
            output_dict: dict, e.g., {
                'waveform': (batch_size, output_channels, segment_samples)
            }
        """
        first_source_type = list(batch_data_dict.keys())[0]
        batch_size = batch_data_dict[first_source_type].shape[0]

        assert (
            batch_size % self.target_sources_num == 0
        ), "Batch size should be \
            evenly divided by target sources number."

        # Get mixture. Sum waveforms all sources.
        stacked_sources = torch.stack(
            [batch_data_dict[source_type] for source_type in self.input_source_types],
            dim=1,
        )
        # input_waveforms: (batch_size, input_sources, input_channels, segment_samples)

        input_waveforms = torch.sum(stacked_sources, dim=1)
        # input_waveforms: (batch_size, input_channels, segment_samples)

        conditions = torch.zeros(batch_size, self.target_sources_num).to(
            input_waveforms.device
        )
        # conditions: (batch_size, target_sources_num)

        target_waveforms = []

        for n in range(batch_size):

            k = n % self.target_sources_num  # source class index
            source_type = self.target_source_types[k]

            target_waveforms.append(batch_data_dict[source_type][n])

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

        target_waveforms = torch.stack(target_waveforms, dim=0)
        # targets: (batch_size, output_channels, segment_samples)

        input_dict = {
            'waveform': input_waveforms,
            'condition': conditions,
        }

        target_dict = {'waveform': target_waveforms}

        return input_dict, target_dict


class AmbisonicBinauralBatchDataPreprocessor(nn.Module):
    def __init__(self, input_source_types: List[str], target_source_types: List[str]):
        r"""Batch data preprocessor. Used for preparing mixtures and targets for
        training. If there are multiple target source types, the waveforms of
        those sources will be stacked along the channel dimension.

        Args:
            input_source_types: List[str], e.g., ['ambisonic']
            target_source_types: List[str], e.g., ['binaural']
        """
        super(AmbisonicBinauralBatchDataPreprocessor, self).__init__()

        self.input_source_types = input_source_types
        self.target_source_types = target_source_types

    def __call__(self, batch_data_dict: Dict) -> List[Dict]:
        r"""Format waveforms and targets for training.

        Args:
            batch_data_dict: dict, e.g., {
                'ambisonic': (batch_size, input_channels, segment_samples),
                'binaural': (batch_size, output_channels, segment_samples),
            }

        Returns:
            input_dict: dict, e.g., {
                'waveform': (batch_size, input_channels, segment_samples),
            }
            output_dict: dict, e.g., {
                'waveform': (batch_size, output_channels, segment_samples)
            }
        """
        input_dict = {'waveform': batch_data_dict['ambisonic']}
        target_dict = {'waveform': batch_data_dict['binaural']}

        return input_dict, target_dict


def get_batch_data_preprocessor_class(batch_data_preprocessor_type: str) -> nn.Module:
    r"""Get batch data preprocessor class.

    Args:
        batch_data_preprocessor_type: str

    Returns:
        nn.Module
    """
    if batch_data_preprocessor_type == 'MixtureTarget':
        return MixtureTargetBatchDataPreprocessor

    elif batch_data_preprocessor_type == 'MixtureTargetConditional':
        return MixtureTargetConditionalBatchDataPreprocessor

    elif batch_data_preprocessor_type == 'AmbisonicBinaural':
        return AmbisonicBinauralBatchDataPreprocessor

    else:
        raise NotImplementedError
