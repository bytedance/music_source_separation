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


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

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

    def forward(self, input_tensor, segment_samples):
        """

        Args:
            input_tensor: (batch_size, channels_num, time_steps)
            segment_samples: int

        Outputs:
            x_pool: (batch_size, channels_num, new_time_steps)
            x: (batch_size, channels_num, time_steps)
        """
        
        batch_size, channels_num, total_samples = input_tensor.shape
        segments_num = total_samples // segment_samples
        
        x = input_tensor.reshape(batch_size, channels_num, segments_num, segment_samples)


        x = x.permute(0, 2, 3, 1)   # (batch_size, segments_num, segment_samples, channels_num)
        x = x.reshape(batch_size * segments_num, segment_samples, channels_num)

        x, _ = self.rnn(x)  # (batch_size * segments_num, segment_samples, channels_num)
        channels_num = x.shape[-1]
        

        x = x.reshape(batch_size, segments_num, segment_samples, channels_num)
        x = x.permute(0, 3, 1, 2)   # (batch_size, channels_num, segments_num, segment_samples)

        x_pool = torch.mean(x, dim=3)
        x = x.reshape(batch_size, channels_num, total_samples)

        return x_pool, x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DecoderBlock, self).__init__()

        momentum = 0.01
        self.segment_samples = stride

        self.conv1 = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride,
            stride=stride,
            padding=0,
            bias=False,
            dilation=1,
        )

        self.bn1 = nn.BatchNorm1d(out_channels, momentum=momentum)

        self.rnn = nn.GRU(
            input_size=out_channels * 2, 
            hidden_size=out_channels // 2, 
            num_layers=1, bias=True, 
            batch_first=True, dropout=0., 
            bidirectional=True
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_gru(self.rnn)

    def forward(self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor):
        r"""Forward data into the module.

        Args:
            torch_tensor: (batch_size, in_feature_maps, downsampled_time_steps)
            concat_tensor: (batch_size, in_feature_maps, time_steps)

        Returns:
            output_tensor: (batch_size, out_feature_maps, time_steps, freq_bins)
        """
        
        x = F.leaky_relu_(self.bn1(self.conv1(input_tensor)), negative_slope=0.01)

        x = torch.cat((x, concat_tensor), dim=1)    # (batch_size, channels_num, total_samples)

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
        
        return x


class LevelRNN(nn.Module, Base):
    def __init__(self, input_channels: int, target_sources_num: int):
        r"""UNet."""
        super(LevelRNN, self).__init__()

        self.input_channels = input_channels
        # self.target_sources_num = target_sources_num

        self.downsample_ratio = 10 ** 4

        self.encoder_block1 = EncoderBlock(in_channels=input_channels, out_channels=16)
        self.encoder_block2 = EncoderBlock(in_channels=16, out_channels=64)
        self.encoder_block3 = EncoderBlock(in_channels=64, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=1024)

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels=1024, stride=10)
        self.decoder_block2 = DecoderBlock(in_channels=1024, out_channels=256, stride=10)
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels=64, stride=10)
        self.decoder_block4 = DecoderBlock(in_channels=64, out_channels=16, stride=10)

        self.after_rnn = EncoderBlock(in_channels=16, out_channels=2)

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        # init_bn(self.bn0)
        # init_layer(self.after_conv2)
        pass

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

        x = mixtures

        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio
            - origin_len
        )
        
        x = F.pad(x, pad=(0, pad_len))

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, segment_samples=10)
        (x2_pool, x2) = self.encoder_block2(x1_pool, segment_samples=10)
        (x3_pool, x3) = self.encoder_block3(x2_pool, segment_samples=10)
        (x4_pool, x4) = self.encoder_block4(x3_pool, segment_samples=10)

        # x_center = self.conv_block7(x6_pool)  # (bs, 384, T / 64, F' / 64)
        x5 = self.decoder_block1(x4_pool, x4)
        x6 = self.decoder_block2(x5, x3)
        x7 = self.decoder_block3(x6, x2)
        x8 = self.decoder_block4(x7, x1)

        _, x = self.after_rnn(x8, segment_samples=10)
        
        x = x[:, :, 0:origin_len]  # (bs, feature_maps, time_steps, freq_bins)

        separated_audio = x

        output_dict = {'waveform': separated_audio}

        return output_dict