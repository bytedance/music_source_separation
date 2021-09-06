import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchlibrosa.stft import STFT, ISTFT, magphase

from bytesep.models.pytorch_modules import (
    Base,
    init_bn,
    init_embedding,
    init_layer,
    act,
    Subband,
)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        condition_size,
        kernel_size,
        activation,
        momentum,
    ):
        super(ConvBlock, self).__init__()

        self.activation = activation
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.beta1 = nn.Linear(condition_size, out_channels, bias=True)
        self.beta2 = nn.Linear(condition_size, out_channels, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_embedding(self.beta1)
        init_embedding(self.beta2)

    def forward(self, x, condition):

        b1 = self.beta1(condition)[:, :, None, None]
        b2 = self.beta2(condition)[:, :, None, None]

        x = act(self.bn1(self.conv1(x)) + b1, self.activation)
        x = act(self.bn2(self.conv2(x)) + b2, self.activation)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        condition_size,
        kernel_size,
        downsample,
        activation,
        momentum,
    ):
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(
            in_channels, out_channels, condition_size, kernel_size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x, condition):
        encoder = self.conv_block(x, condition)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        condition_size,
        kernel_size,
        upsample,
        activation,
        momentum,
    ):
        super(DecoderBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv_block2 = ConvBlock(
            out_channels * 2,
            out_channels,
            condition_size,
            kernel_size,
            activation,
            momentum,
        )

        self.beta1 = nn.Linear(condition_size, out_channels, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_embedding(self.beta1)

    def forward(self, input_tensor, concat_tensor, condition):
        b1 = self.beta1(condition)[:, :, None, None]
        x = act(self.bn1(self.conv1(input_tensor)) + b1, self.activation)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, condition)
        return x


class ConditionalUNet(nn.Module, Base):
    def __init__(self, input_channels, target_sources_num):
        super(ConditionalUNet, self).__init__()

        self.input_channels = input_channels
        condition_size = target_sources_num
        self.output_sources_num = 1

        window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "relu"
        momentum = 0.01

        self.subbands_num = 4
        self.K = 3  # outputs: |M|, cos∠M, sin∠M

        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

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

        self.subband = Subband(subbands_num=self.subbands_num)

        self.encoder_block1 = EncoderBlock(
            in_channels=input_channels * self.subbands_num,
            out_channels=32,
            condition_size=condition_size,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlock(
            in_channels=32,
            out_channels=64,
            condition_size=condition_size,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            condition_size=condition_size,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlock(
            in_channels=128,
            out_channels=256,
            condition_size=condition_size,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlock(
            in_channels=256,
            out_channels=384,
            condition_size=condition_size,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlock(
            in_channels=384,
            out_channels=384,
            condition_size=condition_size,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = ConvBlock(
            in_channels=384,
            out_channels=384,
            condition_size=condition_size,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            condition_size=condition_size,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            condition_size=condition_size,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlock(
            in_channels=384,
            out_channels=256,
            condition_size=condition_size,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            condition_size=condition_size,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            condition_size=condition_size,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlock(
            in_channels=64,
            out_channels=32,
            condition_size=condition_size,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = ConvBlock(
            in_channels=32,
            out_channels=32,
            condition_size=condition_size,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=input_channels
            * self.subbands_num
            * self.output_sources_num
            * self.K,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def feature_maps_to_wav(self, x, sp, sin_in, cos_in, audio_length):

        batch_size, _, time_steps, freq_bins = x.shape

        x = x.reshape(
            batch_size,
            self.output_sources_num,
            self.input_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, output_sources_num, input_channles, K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, output_sources_num, input_channles, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos, out_sin: (batch_size, output_sources_num, input_channles, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag: (batch_size, output_sources_num, input_channles, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, output_sources_num, input_channles, time_steps, freq_bins)

        # Reformat shape to (n, 1, time_steps, freq_bins) for ISTFT.
        shape = (
            batch_size * self.output_sources_num * self.input_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        wav_out = self.istft(out_real, out_imag, audio_length)
        # (batch_size * output_sources_num * input_channels, segments_num)

        # Reshape.
        wav_out = wav_out.reshape(
            batch_size, self.output_sources_num * self.input_channels, audio_length
        )
        # (batch_size, output_sources_num * input_channels, segments_num)

        return wav_out

    def forward(self, input_dict):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        mixture = input_dict['waveform']
        condition = input_dict['condition']

        sp, cos_in, sin_in = self.wav_to_spectrogram_phase(mixture)
        """(batch_size, channels_num, time_steps, freq_bins)"""

        # Batch normalization
        x = sp.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        x = self.subband.analysis(x)

        # UNet
        (x1_pool, x1) = self.encoder_block1(
            x, condition
        )  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(
            x1_pool, condition
        )  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(
            x2_pool, condition
        )  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool, condition
        )  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool, condition
        )  # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool, condition
        )  # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, condition)  # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, condition)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, condition)  # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, condition)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, condition)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, condition)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, condition)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, condition)  # (bs, 32, T, F)
        x = self.after_conv2(x)
        # (batch_size, input_channles * subbands_num * targets_num * k, T, F // subbands_num)

        x = self.subband.synthesis(x)
        # (batch_size, input_channles * targets_num * K, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 1024 -> 1025.
        x = x[:, :, 0:origin_len, :]  # (bs, feature_maps, T, F)

        audio_length = mixture.shape[2]

        separated_audio = self.feature_maps_to_wav(x, sp, sin_in, cos_in, audio_length)
        # separated_audio: (batch_size, output_sources_num * input_channels, segments_num)

        output_dict = {'waveform': separated_audio}

        return output_dict
