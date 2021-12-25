from typing import Dict, Tuple, NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import ISTFT, STFT, magphase

from bytesep.models.pytorch_modules import Base, init_bn, init_layer
from bytesep.models.subband_tools.pqmf import PQMF


class MobileNetConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        activation: str,
        momentum: float,
    ):
        r"""Convolutional block."""
        super(MobileNetConvBlock, self).__init__()

        self.activation = activation
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        r = 1

        self.conv1a = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * r,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        self.conv1b = nn.Conv2d(
            in_channels=out_channels * r,
            out_channels=out_channels * r,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            groups=out_channels * r,
            padding=padding,
            bias=False,
        )

        self.conv1c = nn.Conv2d(
            in_channels=out_channels * r,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        self.conv2a = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * r,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        self.conv2b = nn.Conv2d(
            in_channels=out_channels * r,
            out_channels=out_channels * r,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            groups=out_channels * r,
            padding=padding,
            bias=False,
        )

        self.conv2c = nn.Conv2d(
            in_channels=out_channels * r,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            dilation=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        self.bn1a = nn.BatchNorm2d(in_channels)
        self.bn1b = nn.BatchNorm2d(out_channels * r)
        self.bn1c = nn.BatchNorm2d(out_channels)
        self.bn2a = nn.BatchNorm2d(out_channels)
        self.bn2b = nn.BatchNorm2d(out_channels * r)
        self.bn2c = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_layer(self.conv1a)
        init_layer(self.conv1b)
        init_layer(self.conv1c)
        init_layer(self.conv2a)
        init_layer(self.conv2b)
        init_layer(self.conv2c)
        init_bn(self.bn1a)
        init_bn(self.bn1b)
        init_bn(self.bn1c)
        init_bn(self.bn2a)
        init_bn(self.bn2b)
        init_bn(self.bn2c)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        x = self.conv1a(F.leaky_relu_(self.bn1a(input_tensor), negative_slope=0.01))
        x = self.conv1b(F.leaky_relu_(self.bn1b(x), negative_slope=0.01))
        x = self.conv1c(F.leaky_relu_(self.bn1c(x), negative_slope=0.01))

        if self.is_shortcut:
            origin = self.shortcut(input_tensor) + x
        else:
            origin = input_tensor + x

        x = self.conv2a(F.leaky_relu_(self.bn2a(origin), negative_slope=0.01))
        x = self.conv2b(F.leaky_relu_(self.bn2b(x), negative_slope=0.01))
        x = self.conv2c(F.leaky_relu_(self.bn2c(x), negative_slope=0.01))

        x = origin + x

        return x


class MobileNetEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
        activation: str,
        momentum: float,
    ):
        r"""Encoder block."""
        super(MobileNetEncoderBlock, self).__init__()

        self.conv_block = MobileNetConvBlock(
            in_channels, out_channels, kernel_size, activation, momentum
        )
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        encoder_tensor = self.conv_block(input_tensor)
        # encoder: (batch_size, output_feature_maps, time_steps, freq_bins)

        encoder_pool = F.avg_pool2d(encoder_tensor, kernel_size=self.downsample)
        # encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)

        return encoder_pool, encoder_tensor


class MobileNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
        activation: str,
        momentum: float,
    ):
        r"""Decoder block."""
        super(MobileNetDecoderBlock, self).__init__()

        self.kernel_size = kernel_size
        self.stride = upsample
        self.activation = activation

        self.upsample = torch.nn.Upsample(scale_factor=upsample)
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)

        self.conv_block2 = MobileNetConvBlock(
            out_channels * 2, out_channels, kernel_size, activation, momentum
        )

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        x = self.conv1(F.leaky_relu_(self.bn1(self.upsample(input_tensor))))
        # (batch_size, input_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        output_tensor = self.conv_block2(x)
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return output_tensor


class MobileNet_Subbandtime(nn.Module, Base):
    def __init__(
        self, input_channels: int, output_channels: int, target_sources_num: int
    ):
        r"""Subband waveform UNet."""
        super(MobileNet_Subbandtime, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.target_sources_num = target_sources_num

        window_size = 512  # 2048 // 4
        hop_size = 110  # 441 // 4
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "leaky_relu"
        momentum = 0.01

        self.subbands_num = 4
        self.K = 4  # outputs: |M|, cos∠M, sin∠M, Q

        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

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

        self.encoder_block1 = MobileNetEncoderBlock(
            in_channels=self.input_channels * self.subbands_num,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = MobileNetEncoderBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = MobileNetEncoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = MobileNetEncoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = MobileNetEncoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = MobileNetEncoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(1, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = MobileNetConvBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = MobileNetDecoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(1, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = MobileNetDecoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = MobileNetDecoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = MobileNetDecoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = MobileNetDecoderBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.decoder_block6 = MobileNetDecoderBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = MobileNetConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=self.target_sources_num
            * self.output_channels
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
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * output_channels
        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        waveform = x.reshape(
            batch_size, self.target_sources_num * self.output_channels, audio_length
        )
        # (batch_size, target_sources_num * output_channels, segments_num)

        return waveform

    def forward(self, input_dict):
        r"""Forward data into the module.

        Args:
            input_dict: dict, e.g., {
                waveform: (batch_size, input_channels, segment_samples),
                ...,
            }

        Outputs:
            output_dict: dict, e.g., {
                'waveform': (batch_size, output_channels, segment_samples),
                ...,
            }
        """
        mixtures = input_dict['waveform']
        # (batch_size, input_channels, segment_samples)

        subband_x = self.pqmf.analysis(mixtures)
        # subband_x: (batch_size, input_channels * subbands_num, segment_samples)

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
            int(np.ceil(x.shape[2] / self.time_downsample_ratio))
            * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 257 -> 256
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, input_channels, T, F)
        # x: (batch_size, input_channels * subbands_num, padded_time_steps, freq_bins)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 32, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 64, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool
        )  # x4_pool: (bs, 64, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool
        )  # x5_pool: (bs, 64, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool
        )  # x6_pool: (bs, 64, T / 32, F / 64)
        x_center = self.conv_block7(x6_pool)  # (bs, 64, T / 32, F / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 64, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 64, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 64, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 64, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12)  # (bs, 32, T, F)

        x = self.after_conv2(x)
        # (batch_size, target_sources_num * output_channels * self.K * subbands_num, T, F')

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 256 -> 257.

        x = x[:, :, 0:origin_len, :]
        # (batch_size, target_sources_num * output_channels * self.K * subbands_num, T, F')

        audio_length = subband_x.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_subband_audio = torch.stack(
            [
                self.feature_maps_to_wav(
                    input_tensor=x[:, j :: self.subbands_num, :, :],
                    # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
                    sp=mag[:, j :: self.subbands_num, :, :],
                    # sp: (batch_size, input_channels, T, F')
                    sin_in=sin_in[:, j :: self.subbands_num, :, :],
                    # sin_in: (batch_size, input_channels, T, F')
                    cos_in=cos_in[:, j :: self.subbands_num, :, :],
                    # cos_in: (batch_size, input_channels, T, F')
                    audio_length=audio_length,
                )
                # (batch_size, target_sources_num * output_channels, segments_num)
                for j in range(self.subbands_num)
            ],
            dim=2,
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        # Format for synthesis.
        shape = (
            separated_subband_audio.shape[0],  # batch_size
            self.target_sources_num * self.output_channels * self.subbands_num,
            audio_length,
        )
        separated_subband_audio = separated_subband_audio.reshape(shape)
        # （batch_size, target_sources_num * output_channels * subbands_num, segment_samples)

        separated_audio = self.pqmf.synthesis(separated_subband_audio)
        # (batch_size, input_channles, segment_samples)

        output_dict = {'waveform': separated_audio}

        return output_dict
