from typing import Dict

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


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

    else:
        raise NotImplementedError
