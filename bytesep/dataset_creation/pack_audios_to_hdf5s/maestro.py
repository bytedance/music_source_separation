import argparse
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, NoReturn

import pandas as pd

from bytesep.dataset_creation.pack_audios_to_hdf5s.instruments_solo import (
    write_single_audio_to_hdf5,
)


def read_csv(meta_csv: str) -> Dict:
    r"""Get train & test names from csv.

    Args:
        meta_csv: str

    Returns:
        names_dict: dict, e.g., {
            'train', ['a1.mp3', 'a2.mp3'],
            'test': ['b1.mp3', 'b2.mp3']
        }
    """
    df = pd.read_csv(meta_csv, sep=',')

    names_dict = {}

    for split in ['train', 'test']:
        audio_indexes = df['split'] == split
        audio_names = list(df['audio_filename'][audio_indexes])
        names_dict[split] = audio_names

    return names_dict


def pack_audios_to_hdf5s(args) -> NoReturn:
    r"""Pack (resampled) audio files into hdf5 files to speed up loading.

    Args:
        dataset_dir: str
        split: str, 'train' | 'test'
        hdf5s_dir: str, directory to write out hdf5 files
        sample_rate: int
        channels_num: int
        mono: bool

    Returns:
        NoReturn
    """

    # arguments & parameters
    dataset_dir = args.dataset_dir
    split = args.split
    hdf5s_dir = args.hdf5s_dir
    sample_rate = args.sample_rate
    channels = args.channels
    mono = True if channels == 1 else False

    source_type = "piano"

    # Only pack data for training data.
    assert split == "train"

    # paths
    meta_csv = os.path.join(dataset_dir, 'maestro-v2.0.0.csv')

    os.makedirs(hdf5s_dir, exist_ok=True)

    # Read train & test names.
    names_dict = read_csv(meta_csv)

    audio_names = names_dict['{}'.format(split)]

    params = []

    for audio_index, audio_name in enumerate(audio_names):

        audio_path = os.path.join(dataset_dir, audio_name)

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

    # Uncomment for debug.
    # write_single_audio_to_hdf5(params[0])
    # os._exit(0)

    pack_hdf5s_time = time.time()

    with ProcessPoolExecutor(max_workers=None) as pool:
        # Maximum works on the machine
        pool.map(write_single_audio_to_hdf5, params)

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of the MAESTRO dataset.",
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--hdf5s_dir",
        type=str,
        required=True,
        help="Directory to write out hdf5 files.",
    )
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate.")
    parser.add_argument(
        "--channels", type=int, required=True, help="Use 1 for mono, 2 for stereo."
    )

    # Parse arguments.
    args = parser.parse_args()

    # Pack audios to hdf5 files.
    pack_audios_to_hdf5s(args)
