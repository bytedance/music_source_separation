import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, NoReturn

import h5py
import librosa
import musdb
import numpy as np

from bytesep.utils import float32_to_int16

# Source types of the MUSDB18 dataset.
SOURCE_TYPES = ["vocals", "drums", "bass", "other", "accompaniment"]


def pack_audios_to_hdf5s(args) -> NoReturn:
    r"""Pack (resampled) audio files into hdf5 files to speed up loading.

    Args:
        dataset_dir: str
        subset: str, "train" | "test"
        split: str, "" (100 files) | "train" (86 files) | "valid" (14 files)
        hdf5s_dir: str, directory to write out hdf5 files
        sample_rate: int
        channels_num: int
        mono: bool

    Returns:
        NoReturn
    """

    # arguments & parameters
    dataset_dir = args.dataset_dir
    subset = args.subset
    split = None if args.split == "" else args.split
    hdf5s_dir = args.hdf5s_dir
    sample_rate = args.sample_rate
    channels = args.channels

    mono = True if channels == 1 else False
    source_types = SOURCE_TYPES
    resample_type = "kaiser_fast"

    # Paths
    os.makedirs(hdf5s_dir, exist_ok=True)

    # Dataset of corresponding subset and split.
    mus = musdb.DB(root=dataset_dir, subsets=[subset], split=split)
    print("Subset: {}, Split: {}, Total pieces: {}".format(subset, split, len(mus)))

    params = []  # A list of params for multiple processing.

    for track_index in range(len(mus.tracks)):

        param = (
            dataset_dir,
            subset,
            split,
            track_index,
            source_types,
            mono,
            sample_rate,
            resample_type,
            hdf5s_dir,
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


def write_single_audio_to_hdf5(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""
    (
        dataset_dir,
        subset,
        split,
        track_index,
        source_types,
        mono,
        sample_rate,
        resample_type,
        hdf5s_dir,
    ) = param

    # Dataset of corresponding subset and split.
    mus = musdb.DB(root=dataset_dir, subsets=[subset], split=split)
    track = mus.tracks[track_index]

    # Path to write out hdf5 file.
    hdf5_path = os.path.join(hdf5s_dir, "{}.h5".format(track.name))

    with h5py.File(hdf5_path, "w") as hf:

        hf.attrs.create("audio_name", data=track.name.encode(), dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        for source_type in source_types:

            audio = track.targets[source_type].audio.T
            # (channels_num, audio_samples)

            # Preprocess audio to mono / stereo, and resample.
            audio = preprocess_audio(
                audio, mono, track.rate, sample_rate, resample_type
            )
            # (channels_num, audio_samples) | (audio_samples,)

            hf.create_dataset(
                name=source_type, data=float32_to_int16(audio), dtype=np.int16
            )

        # Mixture
        audio = track.audio.T
        # (channels_num, audio_samples)

        # Preprocess audio to mono / stereo, and resample.
        audio = preprocess_audio(audio, mono, track.rate, sample_rate, resample_type)
        # (channels_num, audio_samples)

        hf.create_dataset(name="mixture", data=float32_to_int16(audio), dtype=np.int16)

    print("{} Write to {}, {}".format(track_index, hdf5_path, audio.shape))


def preprocess_audio(
    audio: np.array, mono: bool, origin_sr: float, sr: float, resample_type: str
) -> np.array:
    r"""Preprocess audio to mono / stereo, and resample.

    Args:
        audio: (channels_num, audio_samples), input audio
        mono: bool
        origin_sr: float, original sample rate
        sr: float, target sample rate
        resample_type: str, e.g., 'kaiser_fast'

    Returns:
        output: ndarray, output audio
    """
    if mono:
        audio = np.mean(audio, axis=0)
        # (audio_samples,)

    output = librosa.core.resample(
        audio, orig_sr=origin_sr, target_sr=sr, res_type=resample_type
    )
    # (audio_samples,) | (channels_num, audio_samples)

    if output.ndim == 1:
        output = output[None, :]
        # (1, audio_samples,)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of the MUSDB18 dataset.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Train subset: 100 pieces; test subset: 50 pieces.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["", "train", "valid"],
        help="Use '' to use all 100 pieces to train. Use 'train' to use 86 \
            pieces for train, and use 'test' to use 14 pieces for valid.",
    )
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

    # Pack audios into hdf5 files.
    pack_audios_to_hdf5s(args)
