import math
from typing import Callable

import torch
import torch.nn as nn
from torchlibrosa.stft import STFT

from bytesep.models.pytorch_modules import Base


def l1(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""L1 loss.

    Args:
        output: torch.Tensor
        target: torch.Tensor

    Returns:
        loss: torch.float
    """
    return torch.mean(torch.abs(output - target))


def l1_wav(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""L1 loss in the time-domain.

    Args:
        output: torch.Tensor
        target: torch.Tensor

    Returns:
        loss: torch.float
    """
    return l1(output, target)


class L1_Wav_L1_Sp(nn.Module, Base):
    def __init__(self):
        r"""L1 loss in the time-domain and L1 loss on the spectrogram."""
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

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        r"""L1 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L1 loss in the time-domain.
        wav_loss = l1_wav(output, target)

        # L1 loss on the spectrogram.
        sp_loss = l1(
            self.wav_to_spectrogram(output, eps=1e-8),
            self.wav_to_spectrogram(target, eps=1e-8),
        )

        # sp_loss /= math.sqrt(self.window_size)
        # sp_loss *= 1.

        # Total loss.
        return wav_loss + sp_loss

        return sp_loss


class L1_Wav_L1_CompressedSp(nn.Module, Base):
    def __init__(self):
        r"""L1 loss in the time-domain and L1 loss on the spectrogram."""
        super(L1_Wav_L1_CompressedSp, self).__init__()

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

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        r"""L1 loss in the time-domain and on the spectrogram.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            loss: torch.float
        """

        # L1 loss in the time-domain.
        wav_loss = l1_wav(output, target)

        output_mag, output_cos, output_sin = self.wav_to_spectrogram_phase(
            output, eps=1e-8
        )
        target_mag, target_cos, target_sin = self.wav_to_spectrogram_phase(
            target, eps=1e-8
        )

        mag_loss = l1(output_mag ** 0.3, target_mag ** 0.3)
        real_loss = l1(output_mag ** 0.3 * output_cos, target_mag ** 0.3 * target_cos)
        imag_loss = l1(output_mag ** 0.3 * output_sin, target_mag ** 0.3 * target_sin)

        total_loss = wav_loss + mag_loss + real_loss + imag_loss

        return total_loss


def get_loss_function(loss_type: str) -> Callable:
    r"""Get loss function.

    Args:
        loss_type: str

    Returns:
        loss function: Callable
    """

    if loss_type == "l1_wav":
        return l1_wav

    elif loss_type == "l1_wav_l1_sp":
        return L1_Wav_L1_Sp()

    elif loss_type == "l1_wav_l1_compressed_sp":
        return L1_Wav_L1_CompressedSp()

    else:
        raise NotImplementedError
