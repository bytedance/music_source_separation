from typing import Dict, List
import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

'''
class LitSourceSeparation(pl.LightningModule):
    def __init__(
        self,
        target_source_type: str,
        model: nn.Module,
        loss_function,
        learning_rate: float,
        lr_lambda,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            target_source_type: str, e.g., 'vocals'
            model: nn.Module
            loss_function: func
            learning_rate: float
            lr_lambda: func
        """
        super().__init__()
        self.target_source_type = target_source_type
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda

    def training_step(self, batch_data_dict: Dict, batch_idx: int) -> float:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'vocals': (batch_size, channels_num, segment_samples),
                'accompaniment': (batch_size, channels_num, segment_samples),
                'mixture': (batch_size, channels_num, segment_samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """

        mixture = batch_data_dict['mixture']
        # (batch_size, channels_num, segment_samples)

        target = batch_data_dict[self.target_source_type]
        # (batch_size, channels_num, segment_samples)

        # Forward.
        self.model.train()

        output = self.model(mixture)['wav']
        # (batch_size, channels_num, segment_samples)

        # Calculate loss.
        loss = self.loss_function(output=output, target=target, mixture=mixture)

        return loss

    def configure_optimizers(self):
        r"""Configure optimizer."""

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
'''

'''
class Musdb18BatchDataPreprocessor:
    def __init__(self, target_source_types):
        self.target_source_types = target_source_types

    def __call__(self, batch_data_dict):
        batch_size = batch_data_dict['mixture'].shape[0]
        target_sources_num = len(self.target_source_types)

        if target_sources_num == 1:
            mixtures = batch_data_dict['mixture']
            targets = batch_data_dict[self.target_source_types[0]]

        else:
            mixtures = batch_data_dict['mixture'].repeat(target_sources_num, 1, 1)
            targets = torch.cat([batch_data_dict[source_type] for source_type in self.target_source_types], dim=0)

        conditions = torch.Tensor(np.zeros((batch_size * self.sources_num, self.condition_size))).to(device)
        # (batch_size, classes_num)

        extra_conditions = torch.Tensor(np.zeros((batch_size * self.sources_num, self.extra_condition_size))).to(device)
        extra_conditions[0 : batch_size, 0] = 1
        extra_conditions[batch_size * 1 : batch_size * 2, 1] = 1
        extra_conditions[batch_size * 2 : batch_size * 3, 2] = 1
        extra_conditions[batch_size * 3 :, 3] = 1

        # from IPython import embed; embed(using=False); os._exit(0)

        return mixtures, sources, conditions, extra_conditions
'''

class Musdb18BatchDataPreprocessor:
    def __init__(self, target_source_types):
        self.target_source_types = target_source_types

    def __call__(self, batch_data_dict):

        batch_size = batch_data_dict['mixture'].shape[0]

        mixtures = batch_data_dict['mixture']
        targets = torch.cat([batch_data_dict[source_type] for source_type in self.target_source_types], dim=1)

        return mixtures, targets


class Musdb18BatchDataPreprocessor2:
    def __init__(self, target_source_types):
        self.target_source_types = target_source_types

    def __call__(self, batch_data_dict):

        batch_size = batch_data_dict['mixture'].shape[0]
        target_sources_num = len(self.target_source_types)

        assert batch_size % target_sources_num == 0

        targets = []
        for n in range(batch_size):
            source_type = self.target_source_types[n % target_sources_num]
            targets.append(batch_data_dict[source_type])

        targets = torch.cat(targets, dim=0)

        return mixtures, targets



class LitSourceSeparation(pl.LightningModule):
    def __init__(
        self,
        # target_source_types: List[str],
        batch_data_preprocessor,
        model: nn.Module,
        loss_function,
        learning_rate: float,
        lr_lambda,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            target_source_type: str, e.g., 'vocals'
            model: nn.Module
            loss_function: func
            learning_rate: float
            lr_lambda: func
        """
        super().__init__()
        # self.target_source_types = target_source_types
        self.batch_data_preprocessor = batch_data_preprocessor
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda

    def training_step(self, batch_data_dict: Dict, batch_idx: int) -> float:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'vocals': (batch_size, channels_num, segment_samples),
                'accompaniment': (batch_size, channels_num, segment_samples),
                'mixture': (batch_size, channels_num, segment_samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        mixtures, targets = self.batch_data_preprocessor(batch_data_dict)
        # mixtures: (batch_size, channels_num, segment_samples)
        # targets: (batch_size, channels_num * target_sources_num, segment_samples)

        # Forward.
        self.model.train()

        outputs = self.model(mixtures)['wav']
        # (batch_size, channels_num, segment_samples)

        # Calculate loss.
        loss = self.loss_function(output=outputs, target=targets, mixture=mixtures)

        return loss

    def configure_optimizers(self):
        r"""Configure optimizer."""

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]


def get_model_class(model_type):
    r"""Get model.

    Args:
        model_type: str, e.g., 'ResUNet143_DecouplePlusInplaceABN'

    Returns:
        nn.Module
    """
    if model_type == 'ResUNet143_DecouplePlusInplaceABN':
        from music_source_separation.models.resunet import (
            ResUNet143_DecouplePlusInplaceABN,
        )

        return ResUNet143_DecouplePlusInplaceABN

    elif model_type == 'UNet':
        from music_source_separation.models.unet import UNet

        return UNet

    else:
        raise NotImplementedError
