from typing import Dict, List, Callable, Any
import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class LitSourceSeparation(pl.LightningModule):
    def __init__(
        self,
        batch_data_preprocessor,
        model: nn.Module,
        loss_function: Callable,
        learning_rate: float,
        lr_lambda: Callable,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            batch_data_preprocessor: object, used for preparing inputs and
                targets for training. E.g., BasicBatchDataPreprocessor.
            model: nn.Module
            loss_function: function
            learning_rate: float
            lr_lambda: function
        """
        super().__init__()
        # self.target_source_types = target_source_types
        self.batch_data_preprocessor = batch_data_preprocessor
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda

    
    def training_step(self, batch_data_dict: Dict, batch_idx: int) -> float:
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_data_dict: e.g. {
                'vocals': (batch_size, channels_num, segment_samples),
                'accompaniment': (batch_size, channels_num, segment_samples),
                'mixture': (batch_size, channels_num, segment_samples)
            }
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        input_dict, target_dict = self.batch_data_preprocessor(batch_data_dict)
        # mixtures: (batch_size, channels_num, segment_samples)
        # targets: e.g., (batch_size, channels_num, segment_samples)

        # Forward.
        self.model.train()

        output_dict = self.model(input_dict)

        outputs = output_dict['waveform']
        # outputs:, e.g, (batch_size, channels_num, segment_samples)

        # if batch_idx == 2:
        #     import soundfile
        #     soundfile.write(file='_zz.wav', data=input_dict['waveform'].data.cpu().numpy()[1, 0], samplerate=44100)
        #     soundfile.write(file='_zz2.wav', data=target_dict['waveform'].data.cpu().numpy()[1, 0], samplerate=44100)
        #     from IPython import embed; embed(using=False); os._exit(0)

        # Calculate loss.
        loss = self.loss_function(
            output=outputs,
            target=target_dict['waveform'],
            mixture=input_dict['waveform'],
        )

        return loss
    

    # def training_step(self, batch_data_dict: Dict, batch_idx: int) -> float:
    #     r"""Forward a mini-batch data to model, calculate loss function, and
    #     train for one step. A mini-batch data is evenly distributed to multiple
    #     devices (if there are) for parallel training.

    #     Args:
    #         batch_data_dict: e.g. {
    #             'vocals': (batch_size, channels_num, segment_samples),
    #             'accompaniment': (batch_size, channels_num, segment_samples),
    #             'mixture': (batch_size, channels_num, segment_samples)
    #         }
    #         batch_idx: int

    #     Returns:
    #         loss: float, loss function of this mini-batch
    #     """

    #     input_dict, target_dict = self.batch_data_preprocessor(batch_data_dict)
    #     # mixtures: (batch_size, channels_num, segment_samples)
    #     # targets: e.g., (batch_size, channels_num, segment_samples)

    #     '''
    #     if batch_idx % 100 == 1:
    #         # from IPython import embed; embed(using=False); os._exit(0)
    #         # print('--', torch.sum(torch.stack([torch.sum(torch.abs(p)) for p in self.model.parameters()])))

    #         self.model.train()
    #         # output_dict = self.model({'waveform': torch.ones(1, 1, 44100).to('cuda')})
    #         output_dict = self.model({'waveform': input_dict['waveform'][0 : 1, :, :]})
    #         # print('---', torch.sum(output_dict['waveform']))
    #         # print('---', torch.sum(torch.abs(input_dict['waveform'][0 : 1, :, :])))
            
    #         # output_dict = self.model({'waveform': torch.ones(1, 1, 44100).to('cuda')})
    #         # print('---', torch.sum(output_dict['waveform']))
    #         # # self.model.train()

    #         # print(torch.sum(torch.abs(self.model.after_conv2.weight)))
    #         # output_dict['waveform'].data.cpu().numpy()
    #         # target_dict['waveform'].data.cpu().numpy()
    #         import soundfile  
    #         import librosa
    #         from pesq import pesq
    #         import numpy as np
    #         soundfile.write(file='_zz0.wav', data=input_dict['waveform'].data.cpu().numpy()[0, 0], samplerate=44100)
    #         soundfile.write(file='_zz1.wav', data=output_dict['waveform'].data.cpu().numpy()[0, 0], samplerate=44100)
    #         soundfile.write(file='_zz2.wav', data=target_dict['waveform'].data.cpu().numpy()[0, 0], samplerate=44100)
    #         # # print(np.sum(np.abs(input_dict['waveform'].data.cpu().numpy()[0, 0])))
    #         mix_a1 = librosa.resample(input_dict['waveform'].data.cpu().numpy()[0, 0], orig_sr=44100, target_sr=16000)
    #         sep_a1 = librosa.resample(output_dict['waveform'].data.cpu().numpy()[0, 0], orig_sr=44100, target_sr=16000)
    #         clean_a1 = librosa.resample(target_dict['waveform'].data.cpu().numpy()[0, 0], orig_sr=44100, target_sr=16000)
    #         pesq_ = pesq(16000, clean_a1, sep_a1, 'wb')
    #         print(batch_idx, pesq_)
    #         # from IPython import embed; embed(using=False); os._exit(0)

    #         # output_dict = self.model({'waveform': torch.ones(1, 1, 44100).to('cuda')})
    #         # print('---', torch.sum(output_dict['waveform']))
    #         # from IPython import embed; embed(using=False); os._exit(0)
    #     '''
    #     # from IPython import embed; embed(using=False); os._exit(0)

    #     if batch_idx % 100 == 1:
    #         # from IPython import embed; embed(using=False); os._exit(0)
    #         # print('--', torch.sum(torch.stack([torch.sum(torch.abs(p)) for p in self.model.parameters()])))

    #         self.model.eval()
    #         # import numpy as np
    #         # random_state = np.random.RandomState(1234)
    #         # output_dict = self.model({'waveform': torch.Tensor(random_state.uniform(-0.1, 0.1, (1, 1, 44100 * 3))).to('cuda')})
    #         # output_dict = self.model({'waveform': torch.ones(1, 1, 44100 * 3).to('cuda')})
    #         output_dict = self.model({'waveform': input_dict['waveform']})
    #         # from IPython import embed; embed(using=False); os._exit(0)

    #         # print('---', torch.sum(torch.abs(input_dict['waveform'][0:1])))
    #         n = 1
    #         # print('---', torch.sum(torch.abs(output_dict['waveform'][n, :, :])))

    #         import soundfile
    #         import librosa
    #         from pesq import pesq
    #         import numpy as np
            
    #         # for n in range(10):
    #         soundfile.write(file='_zz0.wav', data=input_dict['waveform'].data.cpu().numpy()[n, 0], samplerate=44100)
    #         soundfile.write(file='_zz1.wav', data=output_dict['waveform'].data.cpu().numpy()[n, 0], samplerate=44100)
    #         soundfile.write(file='_zz2.wav', data=target_dict['waveform'].data.cpu().numpy()[n, 0], samplerate=44100)
    #         # # print(np.sum(np.abs(input_dict['waveform'].data.cpu().numpy()[0, 0])))
    #         mix_a1 = librosa.resample(input_dict['waveform'].data.cpu().numpy()[n, 0], orig_sr=44100, target_sr=16000)
    #         sep_a1 = librosa.resample(output_dict['waveform'].data.cpu().numpy()[n, 0], orig_sr=44100, target_sr=16000)
    #         clean_a1 = librosa.resample(target_dict['waveform'].data.cpu().numpy()[n, 0], orig_sr=44100, target_sr=16000)
    #         pesq_ = pesq(16000, clean_a1, sep_a1, 'wb')
    #         print(batch_idx, pesq_)
            
    #         # from IPython import embed; embed(using=False); os._exit(0)
    #         # import soundfile
    #         # soundfile.write(file='_zz.wav', data=input_dict['waveform'].data.cpu().numpy()[1, 0], samplerate=44100)
    #         # np.sum(np.abs(input_dict['waveform'].data.cpu().numpy()[1, 0]))

    #     # Forward.
    #     self.model.train()

    #     # input_dict['waveform'] = target_dict['waveform']
    #     output_dict = self.model(input_dict)

    #     outputs = output_dict['waveform'] 
    #     # outputs:, e.g, (batch_size, channels_num, segment_samples)

    #     # if batch_idx == 2: 
    #     #     import soundfile
    #     #     soundfile.write(file='_zz.wav', data=input_dict['waveform'].data.cpu().numpy()[1, 0], samplerate=44100)
    #     #     soundfile.write(file='_zz2.wav', data=target_dict['waveform'].data.cpu().numpy()[1, 0], samplerate=44100)
    #     #     from IPython import embed; embed(using=False); os._exit(0)

    #     # Calculate loss.
    #     loss = self.loss_function(
    #         output=outputs,
    #         target=target_dict['waveform'],
    #         mixture=input_dict['waveform'],
    #     )

    #     '''
    #     if batch_idx % 100 == 1:
    #         # from IPython import embed; embed(using=False); os._exit(0)
    #         # print('--', torch.sum(torch.stack([torch.sum(torch.abs(p)) for p in self.model.parameters()])))

    #         self.model.eval()
    #         # output_dict = self.model({'waveform': torch.ones(2, 1, 44100 * 3).to('cuda')})
    #         output_dict = self.model({'waveform': input_dict['waveform']})
    #         # print('---', torch.sum(input_dict['waveform'][1]))
    #         # print('---', torch.sum(output_dict['waveform'][1, :, :]))

    #         import soundfile
    #         import librosa
    #         from pesq import pesq
    #         import numpy as np
    #         n = 0
    #         # for n in range(10):
    #         soundfile.write(file='_zz0.wav', data=input_dict['waveform'].data.cpu().numpy()[n, 0], samplerate=44100)
    #         soundfile.write(file='_zz1.wav', data=output_dict['waveform'].data.cpu().numpy()[n, 0], samplerate=44100)
    #         soundfile.write(file='_zz2.wav', data=target_dict['waveform'].data.cpu().numpy()[n, 0], samplerate=44100)
    #         # # print(np.sum(np.abs(input_dict['waveform'].data.cpu().numpy()[0, 0])))
    #         mix_a1 = librosa.resample(input_dict['waveform'].data.cpu().numpy()[n, 0], orig_sr=44100, target_sr=16000)
    #         sep_a1 = librosa.resample(output_dict['waveform'].data.cpu().numpy()[n, 0], orig_sr=44100, target_sr=16000)
    #         clean_a1 = librosa.resample(target_dict['waveform'].data.cpu().numpy()[n, 0], orig_sr=44100, target_sr=16000)
    #         pesq_ = pesq(16000, clean_a1, sep_a1, 'wb')
    #         print(batch_idx, pesq_)
            
    #         # from IPython import embed; embed(using=False); os._exit(0)
    #         # import soundfile
    #         # soundfile.write(file='_zz.wav', data=input_dict['waveform'].data.cpu().numpy()[1, 0], samplerate=44100)
    #         # np.sum(np.abs(input_dict['waveform'].data.cpu().numpy()[1, 0]))
        
    #     # print(loss)
    #     # print(torch.sum(torch.abs(input_dict['waveform'])))
    #     '''

    #     return loss
    

    def configure_optimizers(self) -> Any:
        r"""Configure optimizer."""

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
            amsgrad=True,
        )

        scheduler = {
            'scheduler': LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler]


def get_model_class(model_type):
    r"""Get model.

    Args:
        model_type: str, e.g., 'ResUNet143_DecouplePlusInplaceABN'

    Returns:
        nn.Module
    """
    if model_type == 'ResUNet143_DecouplePlusInplaceABN_ISMIR2021':
        from bytesep.models.resunet_ismir2021 import (
            ResUNet143_DecouplePlusInplaceABN_ISMIR2021,
        )
        return ResUNet143_DecouplePlusInplaceABN_ISMIR2021

    elif model_type == 'ResUNet143_DecouplePlusInplaceABNa2':
        from bytesep.models.resunet_ismir2021b import (
            ResUNet143_DecouplePlusInplaceABNa2,
        )
        return ResUNet143_DecouplePlusInplaceABNa2

    elif model_type == 'UNet':
        from bytesep.models.unet import UNet
        return UNet

    elif model_type == 'UNetSubbandTime':
        from bytesep.models.unet_subbandtime import UNetSubbandTime
        return UNetSubbandTime

    # elif model_type == 'UNet2':
    #     from bytesep.models.unet2 import UNet2
    #     return UNet2

    elif model_type == 'ResUNet143_DecouplePlus':
        from bytesep.models.resunet import ResUNet143_DecouplePlus
        return ResUNet143_DecouplePlus

    elif model_type == 'ConditionalUNet':
        from bytesep.models.conditional_unet import ConditionalUNet
        return ConditionalUNet

    elif model_type == 'LevelRNN':
        from bytesep.models.levelrnn import LevelRNN
        return LevelRNN

    elif model_type == 'LevelRNN2':
        from bytesep.models.levelrnn2 import LevelRNN2
        return LevelRNN2

    elif model_type == 'WavUNet':
        from bytesep.models.wavunet import WavUNet
        return WavUNet

    elif model_type == 'WavUNetLevelRNN':
        from bytesep.models.wavunet_levelrnn import WavUNetLevelRNN
        return WavUNetLevelRNN

    else:
        raise NotImplementedError
