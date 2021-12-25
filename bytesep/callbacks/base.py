import logging
import os
from typing import NoReturn

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only


class SaveCheckpointsCallback(pl.Callback):
    def __init__(
        self,
        model: nn.Module,
        checkpoints_dir: str,
        save_step_frequency: int,
    ):
        r"""Callback to save checkpoints every #save_step_frequency steps.

        Args:
            model: nn.Module
            checkpoints_dir: str, directory to save checkpoints
            save_step_frequency: int
        """
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.save_step_frequency = save_step_frequency
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _) -> NoReturn:
        r"""Save checkpoint."""
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0:

            checkpoint_path = os.path.join(
                self.checkpoints_dir, "step={}.pth".format(global_step)
            )

            checkpoint = {'step': global_step, 'model': self.model.state_dict()}

            torch.save(checkpoint, checkpoint_path)
            logging.info("Save checkpoint to {}".format(checkpoint_path))
