import argparse
import os
import pathlib
import pathlib
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import List, NoReturn

import h5py
import librosa
import numpy as np

from bytesep.utils import float32_to_int16


def read_csv(meta_csv):
    df = pd.read_csv(meta_csv, sep=',')
    
    names_dict = {}

    for split in ['train', 'test']:
        audio_indexes = df['split'] == split
        audio_names = list(df['audio_name'][audio_indexes])
        audio_names = ['{}.mp3'.format(pathlib.Path(audio_name).stem) for audio_name in audio_names]
        names_dict['{}_name'.format(split)] = audio_names
        
    return names_dict


def pack_audios_to_hdf5s(args) -> NoReturn:
    """Pack sources of audio files to hdf5 files. Hdf5 files can speed up
    loading and indexing.
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    split = args.split
    source_type = args.source_type
    hdf5s_dir = args.hdf5s_dir
    sample_rate = args.sample_rate
    channels = args.channels
    mono = True if channels == 1 else False

    # Only pack data for training data
    assert split == "train"

    audios_dir = os.path.join(dataset_dir, 'wav48', split)

    os.makedirs(hdf5s_dir, exist_ok=True)

    # audio_names = names_dict['{}_name'.format(split)]

    params = []

    speaker_ids = sorted(os.listdir(audios_dir))

    audio_index = 0

    for speaker_id in speaker_ids:

        speaker_audios_dir = os.path.join(audios_dir, speaker_id)

        audio_names = sorted(os.listdir(speaker_audios_dir))

        for audio_name in audio_names:

            audio_path = os.path.join(speaker_audios_dir, audio_name)

            hdf5_path = os.path.join(
                hdf5s_dir, "{}.h5".format(pathlib.Path(audio_name).stem)
            )

            param = (
                audio_index,
                audio_name,
                source_type,
                audio_path,
                mono,
                sample_rate,
                hdf5_path,
            )
            params.append(param)

            audio_index += 1

    # Uncomment for debug.
    # write_single_audio_to_hdf5(params[0])
    # asdf

    pack_hdf5s_time = time.time()

    with ProcessPoolExecutor(max_workers=None) as pool:
        # Maximum works on the machine
        pool.map(write_single_audio_to_hdf5, params)

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


def write_single_audio_to_hdf5(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        audio_name,
        source_type,
        audio_path,
        mono,
        sample_rate,
        hdf5_path,
    ) = param

    with h5py.File(hdf5_path, "w") as hf:

        hf.attrs.create("audio_name", data=audio_name, dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        audio = load_audio(audio_path=audio_path, mono=mono, sample_rate=sample_rate)

        hf.create_dataset(name=source_type, data=float32_to_int16(audio), dtype=np.int16)

    print('{} Write hdf5 to {}'.format(audio_index, hdf5_path))


def load_audio(audio_path: str, mono: bool, sample_rate: float, offset: float = 0., duration: float = None) -> np.array:
    r"""Load audio.

    Args:
        audio_path: str
        mono: bool
        sample_rate: float
    """
    audio, _ = librosa.core.load(audio_path, sr=sample_rate, mono=mono, offset=offset, duration=duration)
    # (audio_samples,) | (channels_num, audio_samples)

    if audio.ndim == 1:
        audio = audio[None, :]
        # (1, audio_samples,)

    return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # Pack audios to hdf5 files.
    parser_pack_audios = subparsers.add_parser("pack_audios_to_hdf5s")
    parser_pack_audios.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of the Voicebank-Demand dataset.",
    )
    parser_pack_audios.add_argument(
        "--split", type=str, required=True, choices=["train", "test"]
    )
    parser_pack_audios.add_argument(
        "--source_type", type=str, required=True,
    )
    parser_pack_audios.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )
    parser_pack_audios.add_argument(
        "--sample_rate", type=int, required=True, help="Sample rate."
    )
    parser_pack_audios.add_argument(
        "--channels", type=int, required=True, help="Use 1 for mono, 2 for stereo."
    )

    # Parse arguments.
    args = parser.parse_args()

    if args.mode == "pack_audios_to_hdf5s":
        pack_audios_to_hdf5s(args)

    else:
        raise Exception("Incorrect arguments!")
