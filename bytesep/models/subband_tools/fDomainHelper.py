from torchlibrosa.stft import STFT, ISTFT, magphase
import torch
import torch.nn as nn
import numpy as np
from tools.pytorch.modules.pqmf import PQMF


class FDomainHelper(nn.Module):
    def __init__(
        self,
        window_size=2048,
        hop_size=441,
        center=True,
        pad_mode='reflect',
        window='hann',
        freeze_parameters=True,
        subband=None,
        root="/Users/admin/Documents/projects/",
    ):
        super(FDomainHelper, self).__init__()
        self.subband = subband
        if self.subband is None:
            self.stft = STFT(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

            self.istft = ISTFT(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )
        else:
            self.stft = STFT(
                n_fft=window_size // self.subband,
                hop_length=hop_size // self.subband,
                win_length=window_size // self.subband,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

            self.istft = ISTFT(
                n_fft=window_size // self.subband,
                hop_length=hop_size // self.subband,
                win_length=window_size // self.subband,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

        if subband is not None and root is not None:
            self.qmf = PQMF(subband, 64, root)

    def complex_spectrogram(self, input, eps=0.0):
        # [batchsize, samples]
        # return [batchsize, 2, t-steps, f-bins]
        real, imag = self.stft(input)
        return torch.cat([real, imag], dim=1)

    def reverse_complex_spectrogram(self, input, eps=0.0, length=None):
        # [batchsize, 2[real,imag], t-steps, f-bins]
        wav = self.istft(input[:, 0:1, ...], input[:, 1:2, ...], length=length)
        return wav

    def spectrogram(self, input, eps=0.0):
        (real, imag) = self.stft(input.float())
        return torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5

    def spectrogram_phase(self, input, eps=0.0):
        (real, imag) = self.stft(input.float())
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:, channel, :], eps=eps)
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)

        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        return sps, coss, sins

    def spectrogram_phase_to_wav(self, sps, coss, sins, length):
        channels_num = sps.size()[1]
        res = []
        for i in range(channels_num):
            res.append(
                self.istft(
                    sps[:, i : i + 1, ...] * coss[:, i : i + 1, ...],
                    sps[:, i : i + 1, ...] * sins[:, i : i + 1, ...],
                    length,
                )
            )
            res[-1] = res[-1].unsqueeze(1)
        return torch.cat(res, dim=1)

    def wav_to_spectrogram(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size,channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, channel, :], eps=eps))
        output = torch.cat(sp_list, dim=1)
        return output

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.
        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[1]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, channel, :])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(
                self.istft(
                    spectrogram[:, channel : channel + 1, :, :] * cos,
                    spectrogram[:, channel : channel + 1, :, :] * sin,
                    length,
                )
            )

        output = torch.stack(wav_list, dim=1)
        return output

    # todo the following code is not bug free!
    def wav_to_complex_spectrogram(self, input, eps=0.0):
        # [batchsize , channels, samples]
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        res = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            res.append(self.complex_spectrogram(input[:, channel, :], eps=eps))
        return torch.cat(res, dim=1)

    def complex_spectrogram_to_wav(self, input, eps=0.0, length=None):
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        # return  [batchsize, channels, samples]
        channels = input.size()[1] // 2
        wavs = []
        for i in range(channels):
            wavs.append(
                self.reverse_complex_spectrogram(
                    input[:, 2 * i : 2 * i + 2, ...], eps=eps, length=length
                )
            )
            wavs[-1] = wavs[-1].unsqueeze(1)
        return torch.cat(wavs, dim=1)

    def wav_to_complex_subband_spectrogram(self, input, eps=0.0):
        # [batchsize, channels, samples]
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        subspec = self.wav_to_complex_spectrogram(subwav)
        return subspec

    def complex_subband_spectrogram_to_wav(self, input, eps=0.0):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.complex_spectrogram_to_wav(input)
        data = self.qmf.synthesis(subwav)
        return data

    def wav_to_mag_phase_subband_spectrogram(self, input, eps=1e-8):
        """
        :param input:
        :param eps:
        :return:
            loss = torch.nn.L1Loss()
            model = FDomainHelper(subband=4)
            data = torch.randn((3,1, 44100*3))

            sps, coss, sins = model.wav_to_mag_phase_subband_spectrogram(data)
            wav = model.mag_phase_subband_spectrogram_to_wav(sps,coss,sins,44100*3//4)

            print(loss(data,wav))
            print(torch.max(torch.abs(data-wav)))

        """
        # [batchsize, channels, samples]
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        sps, coss, sins = self.wav_to_spectrogram_phase(subwav, eps=eps)
        return sps, coss, sins

    def mag_phase_subband_spectrogram_to_wav(self, sps, coss, sins, length, eps=0.0):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.spectrogram_phase_to_wav(sps, coss, sins, length)
        data = self.qmf.synthesis(subwav)
        return data


if __name__ == "__main__":
    # from thop import profile
    # from thop import clever_format
    # from tools.file.wav import *
    # import time
    #
    # wav = torch.randn((1,2,44100))
    # model = FDomainHelper()

    from tools.file.wav import *

    loss = torch.nn.L1Loss()
    model = FDomainHelper()
    data = torch.randn((3, 1, 44100 * 5))

    sps = model.wav_to_complex_spectrogram(data)
    print(sps.size())
    wav = model.complex_spectrogram_to_wav(sps, 44100 * 5)

    print(loss(data, wav))
    print(torch.max(torch.abs(data - wav)))
