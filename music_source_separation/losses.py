from typing import Callable

from music_source_separation.models.pytorch_modules import Base
from torchlibrosa.stft import STFT
import torch
import torch.nn as nn
import math


def l1(output, target, **kwargs) -> torch.Tensor:
    return torch.mean(torch.abs(output - target))


def l1_wav(output, target, **kwargs) -> torch.Tensor:
    return l1(output, target)


class L1_Wav_L1_Sp(nn.Module, Base):
    def __init__(self):
        super(L1_Wav_L1_Sp, self).__init__()

        self.window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(self, output, target, **kwargs):
        
        # wav_loss = l1_wav(output, target)

        sp_loss = l1(
            self.wav_to_spectrogram(output, eps=1e-8), 
            self.wav_to_spectrogram(target, eps=1e-8)
        )

        # sp_loss /= math.sqrt(self.window_size)
        # sp_loss *= 1.

        # return wav_loss + sp_loss
        return sp_loss


def get_loss_function(loss_type) -> Callable:

    if loss_type == "l1_wav":
        return l1_wav

    elif loss_type == "l1_wav_l1_sp":
        return L1_Wav_L1_Sp()

    else:
        raise NotImplementedError
