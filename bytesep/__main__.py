import argparse
import os
import pathlib
import time
from typing import NoReturn

import soundfile
import torch
import librosa

from bytesep.models.lightning_modules import get_model_class
from bytesep.utils import read_yaml
from bytesep.separate import separate_file, separate_dir

LOCAL_CHECKPOINTS_DIR = os.path.join(pathlib.Path.home(), "bytesep_data")


def download_checkpoints(args) -> NoReturn:
    r"""Download checkpoints and config yaml files from Zenodo."""

    zenodo_dir = "https://zenodo.org/record/5804160/files"
    local_checkpoints_dir = LOCAL_CHECKPOINTS_DIR

    # Download checkpoints.
    checkpoint_names = [
        "mobilenet_subbtandtime_vocals_7.2dB_500k_steps_v2.pth?download=1",
        "mobilenet_subbtandtime_accompaniment_14.6dB_500k_steps_v2.pth?download=1",
        "resunet143_subbtandtime_vocals_8.7dB_500k_steps_v2.pth?download=1",
        "resunet143_subbtandtime_accompaniment_16.4dB_500k_steps_v2.pth?download=1",
    ]

    os.makedirs(local_checkpoints_dir, exist_ok=True)

    for checkpoint_name in checkpoint_names:

        remote_checkpoint_link = os.path.join(zenodo_dir, checkpoint_name)
        local_checkpoint_link = os.path.join(
            local_checkpoints_dir, checkpoint_name.split("?")[0]
        )

        command_str = 'wget -O "{}" "{}"'.format(
            local_checkpoint_link, remote_checkpoint_link
        )
        os.system(command_str)

    # Download and unzip config yaml files.
    remote_zip_scripts_link = os.path.join(zenodo_dir, "train_scripts.zip?download=1")
    local_zip_scripts_path = os.path.join(local_checkpoints_dir, "train_scripts.zip")

    os.system('wget -O "{}" {}'.format(local_zip_scripts_path, remote_zip_scripts_link))
    os.system('unzip "{}" -d {}'.format(local_zip_scripts_path, local_checkpoints_dir))


def get_paths(source_type: str, model_type: str) -> [str, str]:
    r"""Get config_yaml and checkpoint paths.

    Args:
        source_type: str, "vocals" | "accompaniment"
        model_type: str, "MobileNet_Subbandtime" | "ResUNet143_Subbandtime"

    Returns:
        config_yaml: str
        checkpoint_path: str
    """

    local_checkpoints_dir = LOCAL_CHECKPOINTS_DIR

    error_message = "Checkpoint is incomplete, please download again!"

    if model_type == "MobileNet_Subbandtime":

        if source_type == "vocals":

            config_yaml = os.path.join(
                local_checkpoints_dir,
                "train_scripts/musdb18/vocals-accompaniment,mobilenet_subbandtime.yaml",
            )

            checkpoint_path = os.path.join(
                local_checkpoints_dir,
                "mobilenet_subbtandtime_vocals_7.2dB_500k_steps_v2.pth",
            )
            assert os.path.getsize(checkpoint_path) == 4621773, error_message

        elif source_type == "accompaniment":

            config_yaml = os.path.join(
                local_checkpoints_dir,
                "train_scripts/musdb18/accompaniment-vocals,mobilenet_subbandtime.yaml",
            )

            checkpoint_path = os.path.join(
                local_checkpoints_dir,
                "mobilenet_subbtandtime_accompaniment_14.6dB_500k_steps_v2.pth",
            )
            assert os.path.getsize(checkpoint_path) == 4621773, error_message

        else:
            raise NotImplementedError

    elif model_type == "ResUNet143_Subbandtime":

        if source_type == "vocals":

            config_yaml = os.path.join(
                local_checkpoints_dir,
                "train_scripts/musdb18/vocals-accompaniment,resunet_subbandtime.yaml",
            )

            checkpoint_path = os.path.join(
                local_checkpoints_dir,
                "resunet143_subbtandtime_vocals_8.7dB_500k_steps_v2.pth",
            )
            assert os.path.getsize(checkpoint_path) == 414046363, error_message

        elif source_type == "accompaniment":

            config_yaml = os.path.join(
                local_checkpoints_dir,
                "train_scripts/musdb18/accompaniment-vocals,resunet_subbandtime.yaml",
            )

            checkpoint_path = os.path.join(
                local_checkpoints_dir,
                "resunet143_subbtandtime_accompaniment_16.4dB_500k_steps_v2.pth",
            )
            assert os.path.getsize(checkpoint_path) == 414036369, error_message

        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return config_yaml, checkpoint_path


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def separate(args) -> NoReturn:
    r"""Separate an audio file or audio files and write out to a file or directory.

    Args:
        source_type: str
        model_type: str
        audio_path: str, audio file path or directory.
        output_path: str, output audio path or directory.
        scale_volume: bool, set this flag to scale separated audios to maximum value of 1.
        cpu: set this flag to use CPU.
    """

    source_type = args.source_type
    model_type = args.model_type
    audio_path = args.audio_path
    output_path = args.output_path
    scale_volume = args.scale_volume
    cpu = args.cpu

    config_yaml, checkpoint_path = get_paths(source_type, model_type)

    if os.path.isfile(audio_path):

        args = Namespace(
            config_yaml=config_yaml,
            checkpoint_path=checkpoint_path,
            audio_path=audio_path,
            output_path=output_path,
            scale_volume=scale_volume,
            cpu=cpu,
        )

        separate_file(args)

    elif os.path.isdir(audio_path):

        args = Namespace(
            config_yaml=config_yaml,
            checkpoint_path=checkpoint_path,
            audios_dir=audio_path,
            outputs_dir=output_path,
            scale_volume=scale_volume,
            cpu=cpu,
        )

        separate_dir(args)

    else:
        raise Exception("File or directory does not exist!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_download_checkpoints = subparsers.add_parser("download-checkpoints")

    parser_separate = subparsers.add_parser("separate")
    parser_separate.add_argument(
        "--source_type", type=str, default="vocals", choices=["vocals", "accompaniment"]
    )
    parser_separate.add_argument(
        "--model_type",
        type=str,
        default="ResUNet143_Subbandtime",
        choices=["ResUNet143_Subbandtime", "MobileNet_Subbandtime"],
    )
    parser_separate.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="The path or directory of audio(s) to be separated.",
    )
    parser_separate.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path or directory to write out separated audio(s).",
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

    args = parser.parse_args()

    if args.mode == "download-checkpoints":
        download_checkpoints(args)

    elif args.mode == "separate":
        separate(args)

    else:
        raise NotImplementedError
