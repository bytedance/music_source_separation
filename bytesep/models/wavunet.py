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
    init_layer,
    act,
    Subband,
)


class WavUNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, activation, momentum):
        super(WavUNetConvBlock, self).__init__()

        self.activation = activation
        pad = size // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=1,
            dilation=1,
            padding=pad,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels, momentum=momentum)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=1,
            dilation=2,
            padding=pad * 2,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(out_channels, momentum=momentum)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x):
        x = act(self.bn1(self.conv1(x)), self.activation)
        x = act(self.bn2(self.conv2(x)), self.activation)
        return x


class WavUNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(WavUNetEncoderBlock, self).__init__()
        size = 5

        self.conv_block = WavUNetConvBlock(
            in_channels, out_channels, size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block(x)
        encoder_pool = F.avg_pool1d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class WavUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(WavUNetDecoderBlock, self).__init__()

        size = 5
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride,
            stride=stride,
            padding=0,
            output_padding=0,
            bias=False,
            dilation=1,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=momentum)

        self.conv_block2 = WavUNetConvBlock(
            out_channels * 2, out_channels, size, activation, momentum
        )

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn)

    def forward(self, input_tensor, concat_tensor):
        x = act(self.bn1(self.conv1(input_tensor)), self.activation)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        return x


class WavUNet(nn.Module):
    def __init__(self, input_channels, target_sources_num):
        super(WavUNet, self).__init__()
        activation = "relu"

        window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "relu"
        momentum = 0.01

        self.downsample_ratio = 4 ** 6  # This number equals 4^{#encoder_blocks}

        self.bn0 = nn.BatchNorm2d(1, momentum=momentum)

        self.encoder_block1 = WavUNetEncoderBlock(
            in_channels=input_channels,
            out_channels=32,
            downsample=4,
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = WavUNetEncoderBlock(
            in_channels=32,
            out_channels=64,
            downsample=4,
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = WavUNetEncoderBlock(
            in_channels=64,
            out_channels=128,
            downsample=4,
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = WavUNetEncoderBlock(
            in_channels=128,
            out_channels=256,
            downsample=4,
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = WavUNetEncoderBlock(
            in_channels=256,
            out_channels=512,
            downsample=4,
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = WavUNetEncoderBlock(
            in_channels=512,
            out_channels=1024,
            downsample=4,
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = WavUNetConvBlock(
            in_channels=1024,
            out_channels=2048,
            size=5,
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = WavUNetDecoderBlock(
            in_channels=2048,
            out_channels=1024,
            stride=4,
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = WavUNetDecoderBlock(
            in_channels=1024,
            out_channels=512,
            stride=4,
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = WavUNetDecoderBlock(
            in_channels=512,
            out_channels=256,
            stride=4,
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = WavUNetDecoderBlock(
            in_channels=256,
            out_channels=128,
            stride=4,
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = WavUNetDecoderBlock(
            in_channels=128,
            out_channels=64,
            stride=4,
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = WavUNetDecoderBlock(
            in_channels=64,
            out_channels=32,
            stride=4,
            activation=activation,
            momentum=momentum,
        )
        self.after_conv_block1 = WavUNetConvBlock(
            in_channels=32,
            out_channels=32,
            size=5,
            activation=activation,
            momentum=momentum,
        )

        self.after_conv = nn.Conv1d(
            in_channels=32,
            out_channels=input_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv)

    def forward(self, input_dict):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num)}
        """
        mixtures = input_dict['waveform']
        # (batch_size, input_channels, segment_samples)

        x = mixtures

        # Pad waveform to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, pad_len))  # (bs, channels, T)

        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 512, T / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 1024, T / 64)
        x_center = self.conv_block7(x6_pool)  # (bs, 2048, T / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 1024, T / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 512, T / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T)
        x = self.after_conv_block1(x12)  # (bs, 32, T)
        x = self.after_conv(x)  # (bs, channels, T)

        # Reshape
        wav_out = x[:, :, 0:origin_len]

        output_dict = {"waveform": wav_out}

        return output_dict
