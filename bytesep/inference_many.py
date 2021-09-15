import argparse
import os
import pathlib
import time
from typing import NoReturn

import librosa
import numpy as np
import soundfile
import torch

from bytesep.inference import Separator
from bytesep.models.lightning_modules import get_model_class
from bytesep.utils import read_yaml


def inference(args) -> NoReturn:
    r"""Separate all audios in a directory.

    Args:
        config_yaml: str, the config file of a model being trained
        checkpoint_path: str, the path of checkpoint to be loaded
        audios_dir: str, the directory of audios to be separated
        output_dir: str, the directory to write out separated audios
        scale_volume: bool, if True then the volume is scaled to the maximum value of 1.

    Returns:
        NoReturn
    """

    # Arguments & parameters
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    audios_dir = args.audios_dir
    output_dir = args.output_dir
    scale_volume = args.scale_volume
    device = (
        torch.device('cuda')
        if args.cuda and torch.cuda.is_available()
        else torch.device('cpu')
    )

    configs = read_yaml(config_yaml)
    sample_rate = configs['train']['sample_rate']
    input_channels = configs['train']['channels']
    target_source_types = configs['train']['target_source_types']
    target_sources_num = len(target_source_types)
    model_type = configs['train']['model_type']
    mono = input_channels == 1

    segment_samples = int(30 * sample_rate)
    batch_size = 1
    device = "cuda"

    models_contains_inplaceabn = True

    # Need to use torch.distributed if models contain inplace_abn.abn.InPlaceABNSync.
    if models_contains_inplaceabn:

        import torch.distributed as dist

        dist.init_process_group(
            'gloo', init_method='file:///tmp/somefile', rank=0, world_size=1
        )

    print("Using {} for separating ..".format(device))

    # paths
    os.makedirs(output_dir, exist_ok=True)

    # Get model class.
    Model = get_model_class(model_type)

    # Create model.
    model = Model(input_channels=input_channels, target_sources_num=target_sources_num)

    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])

    # Move model to device.
    model.to(device)

    # Create separator.
    separator = Separator(
        model=model,
        segment_samples=segment_samples,
        batch_size=batch_size,
        device=device,
    )

    audio_names = sorted(os.listdir(audios_dir))

    for audio_name in audio_names:
        audio_path = os.path.join(audios_dir, audio_name)

        # Load audio.
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=mono)

        if audio.ndim == 1:
            audio = audio[None, :]

        input_dict = {'waveform': audio}

        # Separate
        separate_time = time.time()

        sep_wav = separator.separate(input_dict)
        # (channels_num, audio_samples)

        print('Separate time: {:.3f} s'.format(time.time() - separate_time))

        # Write out separated audio.
        if scale_volume:
            sep_wav /= np.max(np.abs(sep_wav))

        soundfile.write(file='_zz.wav', data=sep_wav.T, samplerate=sample_rate)

        output_path = os.path.join(
            output_dir, '{}.mp3'.format(pathlib.Path(audio_name).stem)
        )
        os.system('ffmpeg -y -loglevel panic -i _zz.wav "{}"'.format(output_path))
        print('Write out to {}'.format(output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="The config file of a model being trained.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path of checkpoint to be loaded.",
    )
    parser.add_argument(
        "--audios_dir",
        type=str,
        required=True,
        help="The directory of audios to be separated.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to write out separated audios.",
    )
    parser.add_argument(
        '--scale_volume',
        action='store_true',
        default=False,
        help="set to True if separated audios are scaled to the maximum value of 1.",
    )
    parser.add_argument("--cuda", action='store_true', default=True)

    args = parser.parse_args()

    inference(args)
