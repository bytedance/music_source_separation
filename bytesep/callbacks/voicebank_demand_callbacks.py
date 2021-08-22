import glob
import logging
import os
import time
import torch
from typing import List

import librosa
import numpy as np
import pysepm
import pytorch_lightning as pl
import torch.nn as nn
from pesq import pesq
from pytorch_lightning.utilities import rank_zero_only

from bytesep.callbacks.base_callbacks import SaveCheckpointsCallback
from bytesep.inference import Separator
from bytesep.utils import StatisticsContainer, read_yaml


def get_voicebank_demand_callbacks(
    config_yaml: str,
    dataset_dir: str,
    workspace: str,
    checkpoints_dir: str,
    statistics_path: str,
    logger: pl.loggers.TensorBoardLogger,
    model: nn.Module,
    evaluate_device: str,
) -> List[pl.Callback]:
    """Get Voicebank-Demand callbacks of a config yaml.

    Args:
        config_yaml: str
        dataset_dir: str
        workspace: str
        checkpoints_dir: str
        statistics_dir: str
        logger: pl.loggers.TensorBoardLogger
        model: nn.Module
        evaluate_device: str

    Return:
        callbacks: List[pl.Callback]
    """
    configs = read_yaml(config_yaml)
    target_source_types = configs['train']['target_source_types']
    input_channels = configs['train']['channels']
    clean_dir = os.path.join(dataset_dir, configs['evaluate']['test']['clean_dir'])
    noisy_dir = os.path.join(dataset_dir, configs['evaluate']['test']['noisy_dir'])
    sample_rate = configs['train']['sample_rate']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    test_batch_size = configs['evaluate']['batch_size']
    test_segment_seconds = configs['evaluate']['segment_seconds']

    test_segment_samples = int(test_segment_seconds * sample_rate)
    assert len(target_source_types) == 1
    target_source_type = target_source_types[0]
    assert target_source_type == 'speech'

    # save checkpoint callback
    save_checkpoints_callback = SaveCheckpointsCallback(
        model=model,
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    # statistics container
    statistics_container = StatisticsContainer(statistics_path)

    # evaluation callback
    evaluate_test_callback = EvaluationCallback(
        model=model,
        input_channels=input_channels,
        sample_rate=sample_rate,
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    callbacks = [save_checkpoints_callback, evaluate_test_callback]

    return callbacks


def calculate_sdr(ref, est):
    s_true = ref
    s_artif = est - ref
    sdr = 10. * (
        np.log10(np.clip(np.mean(s_true ** 2), 1e-8, np.inf)) \
        - np.log10(np.clip(np.mean(s_artif ** 2), 1e-8, np.inf)))
    return sdr


class EvaluationCallback(pl.Callback):
    def __init__(
        self,
        model: nn.Module,
        input_channels,
        clean_dir,
        noisy_dir,
        sample_rate,
        segment_samples: int,
        batch_size: int,
        device: str,
        evaluate_step_frequency: int,
        logger,
        statistics_container: StatisticsContainer,
    ):
        r"""Callback to evaluate every #save_step_frequency steps.

        Args:
            model: nn.Module
            input_channels: int
            clean_dir: str
            noisy_dir: str
            sample_rate: int
            segment_samples: int, length of segments to be input to a model, e.g., 44100*30
            batch_size, int, e.g., 12
            device: str, e.g., 'cuda'
            evaluate_step_frequency: int, evaluate every #save_step_frequency steps
            logger: object
            statistics_container: StatisticsContainer
        """
        self.model = model
        self.mono = True
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples
        self.evaluate_step_frequency = evaluate_step_frequency
        self.logger = logger
        self.statistics_container = statistics_container

        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir

        self.EVALUATION_SAMPLE_RATE = 16000  # Evaluation sample rate of the
        # Voicebank-Demand task.

        # separator
        self.separator = Separator(model, self.segment_samples, batch_size, device)

    
    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, _) -> None:
        r"""Evaluate losses on a few mini-batches. Losses are only used for
        observing training, and are not final F1 metrics.
        """

        global_step = trainer.global_step

        if global_step % self.evaluate_step_frequency == 0:

            # audio_names = sorted(glob.glob('{}/*.wav'.format(self.clean_dir)))
            audio_names = sorted([audio_name for audio_name in sorted(os.listdir(self.clean_dir)) if audio_name.endswith('.wav')])

            error_str = "Directory {} does not contain audios for evaluation!".format(self.clean_dir)
            assert len(audio_names) > 0, error_str

            pesqs, csigs, cbaks, covls, ssnrs = [], [], [], [], []

            logging.info("--- Step {} ---".format(global_step))
            logging.info("Total {} pieces for evaluation:".format(len(audio_names)))

            eval_time = time.time()

            for n, audio_name in enumerate(audio_names):

                # Load audio.
                clean_path = os.path.join(self.clean_dir, audio_name)
                mixture_path = os.path.join(self.noisy_dir, audio_name)

                mixture, _ = librosa.core.load(
                    mixture_path, sr=self.sample_rate, mono=self.mono
                )

                if mixture.ndim == 1:
                    mixture = mixture[None, :]
                # (channels, audio_length)

                input_dict = {'waveform': mixture}

                # from IPython import embed; embed(using=False); os._exit(0)
                # import soundfile
                # soundfile.write(file='_zz.wav', data=mixture[0], samplerate=self.sample_rate)

                # separate
                sep_wav = self.separator.separate(input_dict)
                # (channels, audio_length)

                # Target
                clean, _ = librosa.core.load(
                    clean_path, sr=self.EVALUATION_SAMPLE_RATE, mono=self.mono
                )

                # to mono
                sep_wav = np.squeeze(sep_wav)

                # Resample for evaluation.
                sep_wav = librosa.resample(
                    sep_wav,
                    orig_sr=self.sample_rate,
                    target_sr=self.EVALUATION_SAMPLE_RATE,
                )

                sep_wav = librosa.util.fix_length(sep_wav, size=len(clean), axis=0)
                # (channels, audio_length)

                # Evaluate metrics
                pesq_ = pesq(self.EVALUATION_SAMPLE_RATE, clean, sep_wav, 'wb')
                # pesq_ = pesq(self.EVALUATION_SAMPLE_RATE, clean, sep_wav, 'nb')

                (csig, cbak, covl) = pysepm.composite(
                    clean, sep_wav, self.EVALUATION_SAMPLE_RATE
                )

                ssnr = pysepm.SNRseg(clean, sep_wav, self.EVALUATION_SAMPLE_RATE)

                pesqs.append(pesq_)
                csigs.append(csig)
                cbaks.append(cbak)
                covls.append(covl)
                ssnrs.append(ssnr)
                print(
                    '{}, {}, PESQ: {:.3f}, CSIG: {:.3f}, CBAK: {:.3f}, COVL: {:.3f}, SSNR: {:.3f}'.format(
                        n, audio_name, pesq_, csig, cbak, covl, ssnr
                    )
                )
                # from IPython import embed; embed(using=False); os._exit(0)

                # if n == 10:
                #     break

            logging.info("-----------------------------")
            logging.info('Avg PESQ: {:.3f}'.format(np.mean(pesqs)))
            logging.info('Avg CSIG: {:.3f}'.format(np.mean(csigs)))
            logging.info('Avg CBAK: {:.3f}'.format(np.mean(cbaks)))
            logging.info('Avg COVL: {:.3f}'.format(np.mean(covls)))
            logging.info('Avg SSNR: {:.3f}'.format(np.mean(ssnrs)))

            logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

            statistics = {"pesq": np.mean(pesqs)}
            self.statistics_container.append(global_step, statistics, 'test')
            self.statistics_container.dump()
    

    
    # @rank_zero_only
    # def on_batch_end(self, trainer: pl.Trainer, _) -> None:
    #     r"""Evaluate losses on a few mini-batches. Losses are only used for
    #     observing training, and are not final F1 metrics.
    #     """

    #     global_step = trainer.global_step

    #     if global_step % self.evaluate_step_frequency == 0:

    #         # audio_names = sorted(glob.glob('{}/*.wav'.format(self.clean_dir)))
    #         audio_names = sorted([audio_name for audio_name in sorted(os.listdir(self.clean_dir)) if audio_name.endswith('.wav')])

    #         error_str = "Directory {} does not contain audios for evaluation!".format(self.clean_dir)
    #         assert len(audio_names) > 0, error_str

    #         pesqs, csigs, cbaks, covls, ssnrs = [], [], [], [], []

    #         logging.info("--- Step {} ---".format(global_step))
    #         logging.info("Total {} pieces for evaluation:".format(len(audio_names)))

    #         eval_time = time.time()

    #         for n, audio_name in enumerate(audio_names):
    #             # print(torch.sum(torch.stack([torch.sum(torch.abs(p)) for p in self.model.parameters()])))
    #             # from IPython import embed; embed(using=False); os._exit(0)
    #             # print(torch.sum(self.model.decoder_block6.bn1.weight))
    #             # print(torch.sum(self.model.bn0.running_mean))

    #             # self.model.train()
    #             # self.model.eval()
    #             # output_dict = self.model({'waveform': torch.ones(1, 1, 44100).to('cuda')})
    #             # print('-', torch.sum(output_dict['waveform']))
    #             # output_dict = self.model({'waveform': torch.ones(1, 1, 44100).to('cuda')})
    #             # print('-', torch.sum(output_dict['waveform']))
    #             # self.model.train()

                
    #             h5_path = '/home/tiger/my_code_2019.12-/python/bytesep/workspaces/music_source_separation/hdf5s/voicebank-demand/sr=44100_ch=1/train/p226_002.h5'

    #             import h5py
    #             from music_source_separation.utils import int16_to_float32
    #             hf = h5py.File(h5_path, 'r')
    #             clean = int16_to_float32(hf['speech'][0, :])
    #             noise = int16_to_float32(hf['noise'][0, :])
    #             mixture = int16_to_float32(hf['speech'][0, :]) + int16_to_float32(hf['noise'][0, :])
                
    #             '''
    #             if n == 1:
    #                 # Load audio.
    #                 clean_path = os.path.join(self.clean_dir, audio_name)
    #                 mixture_path = os.path.join(self.noisy_dir, audio_name)

    #                 mixture2, _ = librosa.core.load(mixture_path, sr=44100, mono=True)
    #                 clean2, _ = librosa.core.load(clean_path, sr=44100, mono=True)

    #                 from bytesep.utils import int16_to_float32, float32_to_int16
    #                 noise2 = int16_to_float32(float32_to_int16(mixture2 - clean2))
    #                 clean2 = int16_to_float32(float32_to_int16(clean2))
                    
    #                 mixture2 = clean2 + noise2

    #                 # from IPython import embed; embed(using=False); os._exit(0)
    #                 # import soundfile
    #                 # soundfile.write(file='_zz1.wav', data=clean, samplerate=44100)
    #                 # soundfile.write(file='_zz2.wav', data=clean2, samplerate=44100)

    #                 mixture = mixture2
    #                 clean = clean2
    #             '''

    #             '''    
    #             # Load audio.
    #             clean_path = os.path.join(self.clean_dir, audio_name)
    #             mixture_path = os.path.join(self.noisy_dir, audio_name)

    #             mixture2, _ = librosa.core.load(mixture_path, sr=44100, mono=True)
    #             clean2, _ = librosa.core.load(clean_path, sr=44100, mono=True)

    #             from IPython import embed; embed(using=False); os._exit(0)
    #             # import soundfile
    #             # soundfile.write(file='_zz1.wav', data=clean, samplerate=44100)
    #             # soundfile.write(file='_zz2.wav', data=clean2, samplerate=44100)

    #             mixture = mixture2
    #             clean = clean2
    #             '''

                    

    #             # Target
    #             # clean, _ = librosa.core.load(clean_path, sr=16000, mono=True)

    #             # mixture = librosa.util.fix_length(mixture, size=44100 * 3, axis=0)

    #             if mixture.ndim == 1:
    #                 mixture = mixture[None, None, :]
    #             # (channels, audio_length)

    #             # if n == 1:
    #             #     from IPython import embed; embed(using=False); os._exit(0)
    #             #     import soundfile
    #             #     soundfile.write(file='_zz.wav', data=mixture[0, 0], samplerate=44100)
                
    #             # random_state = np.random.RandomState(1234)
    #             # input_dict = {'waveform': torch.Tensor(random_state.uniform(-0.1, 0.1, (1, 1, 44100 * 3))).to('cuda')}
    #             input_dict = {'waveform': torch.Tensor(mixture).to('cuda')}
    #             # input_dict = {'waveform': torch.ones(1, 1, 44100 * 3).to('cuda')}
    #             # from IPython import embed; embed(using=False); os._exit(0)

    #             self.model.eval()
    #             # self.model.train()
    #             output_dict = self.model(input_dict)
    #             # print(torch.sum(torch.abs(torch.Tensor(mixture).to('cuda'))))
    #             # print(torch.sum(torch.abs(output_dict['waveform'])))

    #             # print(torch.sum(torch.abs(self.model.after_conv2.weight)))
                
    #             sep_wav = output_dict['waveform'].data.cpu().numpy()[0, 0]
    #             # print(np.sum(np.abs(sep_wav)))

    #             sep_wav = librosa.resample(sep_wav, orig_sr=44100, target_sr=16000)

    #             # sep_wav = librosa.util.fix_length(sep_wav, size=44100 * 3, axis=0)
    #             # (channels, audio_length)

    #             # Evaluate metrics
    #             clean = librosa.resample(clean, orig_sr=44100, target_sr=16000)

    #             pesq_ = pesq(16000, clean, sep_wav, 'wb')
    #             # pesq_ = pesq(self.EVALUATION_SAMPLE_RATE, clean, sep_wav, 'nb')
    #             print(pesq_)

    #             sdr = calculate_sdr(clean, sep_wav)
    #             print('=== SDR', sdr)

    #             # from IPython import embed; embed(using=False); os._exit(0)
                
    #             if n == 5:
    #                 break

    #         logging.info("-----------------------------")
    #         logging.info('Avg PESQ: {:.3f}'.format(np.mean(pesqs)))
    #         logging.info('Avg CSIG: {:.3f}'.format(np.mean(csigs)))
    #         logging.info('Avg CBAK: {:.3f}'.format(np.mean(cbaks)))
    #         logging.info('Avg COVL: {:.3f}'.format(np.mean(covls)))
    #         logging.info('Avg SSNR: {:.3f}'.format(np.mean(ssnrs)))

    #         logging.info("Evlauation time: {:.3f}".format(time.time() - eval_time))

    #         statistics = {"pesq": np.mean(pesqs)}
    #         self.statistics_container.append(global_step, statistics, 'test')
    #         self.statistics_container.dump()
    