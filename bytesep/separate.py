import argparse
import os
import pathlib
import time
from typing import NoReturn

import numpy as np
import soundfile
import torch

from bytesep.models.lightning_modules import get_model_class
from bytesep.separator import Separator
from bytesep.utils import load_audio, read_yaml


def init_abn() -> NoReturn:
    # Need to use torch.distributed if models contain inplace_abn.abn.InPlaceABNSync.
    import torch.distributed as dist

    dist.init_process_group(
        'gloo', init_method='file:///tmp/somefile', rank=0, world_size=1
    )


def build_separator(config_yaml: str, checkpoint_path: str, device: str) -> Separator:
    r"""Build separator.

    Args:
        config_yaml: str
        checkpoint_path: str
        device: "cuda" | "cpu"

    Returns:
        separator: Separator
    """

    # Read config file.
    configs = read_yaml(config_yaml)
    sample_rate = configs['train']['sample_rate']
    input_channels = configs['train']['input_channels']
    output_channels = configs['train']['output_channels']
    target_source_types = configs['train']['target_source_types']
    target_sources_num = len(target_source_types)
    model_type = configs['train']['model_type']

    segment_seconds = 30
    segment_samples = int(segment_seconds * sample_rate)
    batch_size = 1

    print("Using {} for separating ..".format(device))

    models_contains_inplaceabn = False

    if models_contains_inplaceabn:
        init_abn(models_contains_inplaceabn)

    # Get model class.
    Model = get_model_class(model_type)

    # Create model.
    model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        target_sources_num=target_sources_num,
    )

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

    return separator


def match_audio_channels(audio: np.array, input_channels: int) -> np.array:
    r"""Match input audio to correct channels.

    Args:
        audio: (audio_channels, audio_segments)
        input_channels: int

    Returns:
        (input_channels, audio_segments)
    """

    audio_channels = audio.shape[0]

    if audio_channels == input_channels:
        return audio

    elif audio_channels == 2 and input_channels == 1:
        return np.mean(audio, axis=0)[None, :]

    elif audio_channels == 1 and input_channels == 2:
        return np.tile(audio, (2, 1))

    else:
        raise NotImplementedError


def separate_file(args) -> NoReturn:
    r"""Separate a single file.

    Args:
        config_yaml: str, the config file of a model being trained.
        checkpoint_path: str, the path of checkpoint to be loaded.
        audio_path: str, path of file to be separated.
        output_path: str, path of separated file to be written.
        scale_volume: bool
        cpu: bool

    Returns:
        NoReturn
    """

    # Arguments & parameters
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    output_path = args.output_path
    scale_volume = args.scale_volume
    cpu = args.cpu

    if cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Read yaml files.
    configs = read_yaml(config_yaml)
    sample_rate = configs['train']['sample_rate']
    input_channels = configs['train']['input_channels']

    # Build Separator.
    separator = build_separator(config_yaml, checkpoint_path, device)

    # paths
    if os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load audio.
    audio = load_audio(audio_path=audio_path, mono=False, sample_rate=sample_rate)
    # audio: (input_channels, audio_samples)

    audio = match_audio_channels(audio, input_channels)
    # audio: (input_channels, audio_samples)

    input_dict = {'waveform': audio}

    # Separate
    separate_time = time.time()

    sep_audio = separator.separate(input_dict)
    # (input_channels, audio_samples)

    print('Separate time: {:.3f} s'.format(time.time() - separate_time))

    # Write out separated audio.
    if scale_volume:
        sep_audio /= np.max(np.abs(sep_audio))

    # Write out separated audio.
    tmp_wav_path = output_path + ".wav"
    soundfile.write(file=tmp_wav_path, data=sep_audio.T, samplerate=sample_rate)

    os.system(
        'ffmpeg -y -loglevel panic -i "{}" "{}"'.format(tmp_wav_path, output_path)
    )
    os.system('rm "{}"'.format(tmp_wav_path))

    print('Write out to {}'.format(output_path))


def separate_dir(args) -> NoReturn:
    r"""Separate all audios in a directory.

    Args:
        config_yaml: str, the config file of a model being trained.
        checkpoint_path: str, the path of checkpoint to be loaded.
        audios_dir: str, the directory of audios to be separated.
        output_dir: str, the directory to write out separated audios.
        scale_volume: bool, if True then the volume is scaled to the maximum value of 1.
        cpu: bool

    Returns:
        NoReturn
    """

    # Arguments & parameters
    config_yaml = args.config_yaml
    checkpoint_path = args.checkpoint_path
    audios_dir = args.audios_dir
    outputs_dir = args.outputs_dir
    scale_volume = args.scale_volume
    cpu = args.cpu

    if cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Read yaml files.
    configs = read_yaml(config_yaml)
    sample_rate = configs['train']['sample_rate']

    # Build separator.
    separator = build_separator(config_yaml, checkpoint_path, device)

    # paths
    os.makedirs(outputs_dir, exist_ok=True)

    audio_names = sorted(os.listdir(audios_dir))
    audios_num = len(audio_names)

    for n, audio_name in enumerate(audio_names):

        audio_path = os.path.join(audios_dir, audio_name)

        # Load audio.
        audio = load_audio(audio_path=audio_path, mono=False, sample_rate=sample_rate)

        input_dict = {'waveform': audio}

        # Separate
        separate_time = time.time()

        sep_audio = separator.separate(input_dict)
        # (input_channels, audio_samples)

        print('Separate time: {:.3f} s'.format(time.time() - separate_time))

        # Write out separated audio.
        if scale_volume:
            sep_audio /= np.max(np.abs(sep_audio))

        output_path = os.path.join(
            outputs_dir, '{}.mp3'.format(pathlib.Path(audio_name).stem)
        )

        tmp_wav_path = '{}.wav'.format(output_path)
        soundfile.write(file=tmp_wav_path, data=sep_audio.T, samplerate=sample_rate)

        os.system(
            'ffmpeg -y -loglevel panic -i "{}" "{}"'.format(tmp_wav_path, output_path)
        )
        os.system('rm "{}"'.format(tmp_wav_path))
        print('{} / {}, Write out to {}'.format(n, audios_num, output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_separate = subparsers.add_parser("separate_file")
    parser_separate.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="The config file of a model being trained.",
    )
    parser_separate.add_argument(
        "--checkpoint_path", type=str, required=True, help="Checkpoint path."
    )
    parser_separate.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="The path of audio to be separated.",
    )
    parser_separate.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to write out separated audio.",
    )
    parser_separate.add_argument(
        '--scale_volume',
        action='store_true',
        default=False,
        help="Set this flag to scale separated audios to maximum value of 1.",
    )
    parser_separate.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help="Set this flag to use CPU.",
    )

    parser_separate_dir = subparsers.add_parser("separate_dir")
    parser_separate_dir.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="The config file of a model being trained.",
    )
    parser_separate_dir.add_argument(
        "--checkpoint_path", type=str, required=True, help="Checkpoint path."
    )
    parser_separate_dir.add_argument(
        "--audios_dir",
        type=str,
        required=True,
        help="The directory of audios to be separated.",
    )
    parser_separate_dir.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="The directory to write out separated audios.",
    )
    parser_separate_dir.add_argument(
        '--scale_volume',
        action='store_true',
        default=False,
        help="Set this flag to scale separated audios to maximum value of 1.",
    )
    parser_separate_dir.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help="Set this flag to use CPU.",
    )

    args = parser.parse_args()

    if args.mode == "separate_file":
        separate_file(args)

    elif args.mode == "separate_dir":
        separate_dir(args)

    else:
        raise NotImplementedError
