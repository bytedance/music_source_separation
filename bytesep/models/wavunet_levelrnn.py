import math
from typing import List, Tuple, NoReturn, Dict

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

from bytesep.models.wavunet import WavUNetEncoderBlock, WavUNetDecoderBlock, WavUNetConvBlock


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class RNN(nn.Module):
    def __init__(self, in_channels, out_channels, segment_samples):
        super(RNN, self).__init__()

        self.segment_samples = segment_samples

        self.rnn = nn.GRU(
            input_size=in_channels, 
            hidden_size=out_channels // 2, 
            num_layers=1, bias=True, 
            batch_first=True, dropout=0., 
            bidirectional=True
        )

        self.init_weights()

    def init_weights(self):
        init_gru(self.rnn)

    def forward(self, input_tensor):
        """

        Args:
            input_tensor: (batch_size, channels_num, time_steps)
            segment_samples: int

        Outputs:
            x_pool: (batch_size, channels_num, new_time_steps)
            x: (batch_size, channels_num, time_steps)
        """
        x = input_tensor
        origin_len = x.shape[2]
        pad_len = (int(np.ceil(x.shape[2] / self.segment_samples)) * self.segment_samples - origin_len)
        
        x = F.pad(x, pad=(0, pad_len))
        
        batch_size, channels_num, total_samples = x.shape
        segments_num = total_samples // self.segment_samples
        
        x = x.reshape(batch_size, channels_num, segments_num, self.segment_samples)

        x = x.permute(0, 2, 3, 1)   # (batch_size, segments_num, segment_samples, channels_num)
        x = x.reshape(batch_size * segments_num, self.segment_samples, channels_num)

        x, _ = self.rnn(x)  # (batch_size * segments_num, segment_samples, channels_num)
        channels_num = x.shape[-1]
        
        x = x.reshape(batch_size, segments_num, self.segment_samples, channels_num)
        x = x.permute(0, 3, 1, 2)   # (batch_size, channels_num, segments_num, segment_samples)

        x = x.reshape(batch_size, channels_num, total_samples)
        x = x[:, :, 0 : origin_len]

        output = x + input_tensor

        return output


class WavUNetLevelRNN(nn.Module):
    def __init__(self, input_channels, target_sources_num):
        super(WavUNetLevelRNN, self).__init__()
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

        self.encoder_rnn1 = RNN(in_channels=32, out_channels=32, segment_samples=100)
        self.encoder_rnn2 = RNN(in_channels=64, out_channels=64, segment_samples=100)
        self.encoder_rnn3 = RNN(in_channels=128, out_channels=128, segment_samples=100)
        self.encoder_rnn4 = RNN(in_channels=256, out_channels=256, segment_samples=100)
        self.encoder_rnn5 = RNN(in_channels=512, out_channels=512, segment_samples=100)
        self.encoder_rnn6 = RNN(in_channels=1024, out_channels=1024, segment_samples=100)
        self.encoder_rnn7 = RNN(in_channels=2048, out_channels=2048, segment_samples=100)
        self.decoder_rnn1 = RNN(in_channels=1024, out_channels=1024, segment_samples=100)
        self.decoder_rnn2 = RNN(in_channels=512, out_channels=512, segment_samples=100)
        self.decoder_rnn3 = RNN(in_channels=256, out_channels=256, segment_samples=100)
        self.decoder_rnn4 = RNN(in_channels=128, out_channels=128, segment_samples=100)
        self.decoder_rnn5 = RNN(in_channels=64, out_channels=64, segment_samples=100)
        self.decoder_rnn6 = RNN(in_channels=32, out_channels=32, segment_samples=100)

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
        x1_pool = self.encoder_rnn1(x1_pool)

        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4)
        x2_pool = self.encoder_rnn2(x2_pool)

        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8)
        x3_pool = self.encoder_rnn3(x3_pool)

        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16)
        x4_pool = self.encoder_rnn4(x4_pool)

        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 512, T / 32)
        x5_pool = self.encoder_rnn5(x5_pool)

        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 1024, T / 64)
        x6_pool = self.encoder_rnn6(x6_pool)

        x_center = self.conv_block7(x6_pool)  # (bs, 2048, T / 64)
        x_center = self.encoder_rnn7(x_center)

        x7 = self.decoder_block1(x_center, x6)  # (bs, 1024, T / 32)
        x7 = self.decoder_rnn1(x7)

        x8 = self.decoder_block2(x7, x5)  # (bs, 512, T / 16)
        x8 = self.decoder_rnn2(x8)

        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8)
        x9 = self.decoder_rnn3(x9)

        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4)
        x10 = self.decoder_rnn4(x10)

        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2)
        x11 = self.decoder_rnn5(x11)

        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T)
        x12 = self.decoder_rnn6(x12)

        x = self.after_conv_block1(x12)  # (bs, 32, T)
        x = self.after_conv(x)  # (bs, channels, T)

        # Reshape
        wav_out = x[:, :, 0:origin_len]
        
        output_dict = {"waveform": wav_out}
        
        return output_dict