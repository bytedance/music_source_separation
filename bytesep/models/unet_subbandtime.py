from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import ISTFT, STFT, magphase

from bytesep.models.pytorch_modules import Base, init_bn, init_layer
from bytesep.models.subband_tools.pqmf import PQMF
from bytesep.models.unet import ConvBlock, DecoderBlock, EncoderBlock


class UNetSubbandTime(nn.Module, Base):
    def __init__(self, input_channels: int, target_sources_num: int):
        r"""Subband waveform UNet."""
        super(UNetSubbandTime, self).__init__()

        self.input_channels = input_channels
        self.target_sources_num = target_sources_num

        window_size = 512  # 2048 // 4
        hop_size = 110  # 441 // 4
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "leaky_relu"
        momentum = 0.01

        self.subbands_num = 4
        self.K = 3  # outputs: |M|, cos∠M, sin∠M

        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.pqmf = PQMF(
            N=self.subbands_num,
            M=64,
            project_root='bytesep/models/subband_tools/filters',
        )

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlock(
            in_channels=input_channels * self.subbands_num,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlock(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = ConvBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlock(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.decoder_block6 = DecoderBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=target_sources_num
            * input_channels
            * self.K
            * self.subbands_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * input_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
            sin_in: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)
            cos_in: (batch_size, target_sources_num * input_channels, time_steps, freq_bins)

        Outputs:
            waveform: (batch_size, target_sources_num * input_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.input_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, input_channles, K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, input_channles, time_steps, freq_bins)

        # Reformat shape to (n, 1, time_steps, freq_bins) for ISTFT.
        shape = (
            batch_size * self.target_sources_num * self.input_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * input_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.input_channels, audio_length
        )
        # (batch_size, target_sources_num * input_channels, segments_num)

        return waveform

    def forward(self, input_dict: Dict) -> Dict:
        """Forward data into the module.

        Args:
            input_dict: dict, e.g., {
                waveform: (batch_size, input_channels, segment_samples),
                ...,
            }

        Outputs:
            output_dict: dict, e.g., {
                'waveform': (batch_size, input_channels, segment_samples),
                ...,
            }
        """
        mixtures = input_dict['waveform']
        # (batch_size, input_channels, segment_samples)

        if self.subbands_num > 1:
            subband_x = self.pqmf.analysis(mixtures)
            # -- subband_x: (batch_size, input_channels * subbands_num, segment_samples)
            # -- subband_x: (batch_size, subbands_num * input_channels, segment_samples)
        else:
            subband_x = mixtures

        # from IPython import embed; embed(using=False); os._exit(0)
        # import soundfile
        # soundfile.write(file='_zz.wav', data=subband_x.data.cpu().numpy()[0, 2], samplerate=11025)

        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(subband_x)
        # mag, cos_in, sin_in: (batch_size, input_channels * subbands_num, time_steps, freq_bins)

        # Batch normalize on individual frequency bins.
        x = mag.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # (batch_size, input_channels * subbands_num, time_steps, freq_bins)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F' / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F' / 4)
        (x3_pool, x3) = self.encoder_block3(
            x2_pool
        )  # x3_pool: (bs, 128, T / 8, F' / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool
        )  # x4_pool: (bs, 256, T / 16, F' / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool
        )  # x5_pool: (bs, 384, T / 32, F' / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool
        )  # x6_pool: (bs, 384, T / 64, F' / 64)
        x_center = self.conv_block7(x6_pool)  # (bs, 384, T / 64, F' / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T / 32, F' / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T / 16, F' / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F' / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F' / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F' / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F')
        x = self.after_conv_block1(x12)  # (bs, 32, T, F')

        x = self.after_conv2(x)
        # (batch_size, subbands_num * target_sources_num * input_channles * self.K, T, F')

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257.

        x = x[:, :, 0:origin_len, :]
        # (batch_size, subbands_num * target_sources_num * input_channles * self.K, T, F')

        audio_length = subband_x.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        C1 = x.shape[1] // self.subbands_num
        C2 = mag.shape[1] // self.subbands_num

        separated_subband_audio = torch.cat(
            [
                self.feature_maps_to_wav(
                    input_tensor=x[:, j * C1 : (j + 1) * C1, :, :],
                    sp=mag[:, j * C2 : (j + 1) * C2, :, :],
                    sin_in=sin_in[:, j * C2 : (j + 1) * C2, :, :],
                    cos_in=cos_in[:, j * C2 : (j + 1) * C2, :, :],
                    audio_length=audio_length,
                )
                for j in range(self.subbands_num)
            ],
            dim=1,
        )
        # （batch_size, subbands_num * target_sources_num * input_channles, segment_samples)

        if self.subbands_num > 1:
            separated_audio = self.pqmf.synthesis(separated_subband_audio)
            # (batch_size, target_sources_num * input_channles, segment_samples)
        else:
            separated_audio = separated_subband_audio

        output_dict = {'waveform': separated_audio}

        return output_dict
