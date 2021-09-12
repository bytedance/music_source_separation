import argparse
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, NoReturn

import h5py
import numpy as np

from bytesep.utils import float32_to_int16, load_audio


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

    # Only pack data for training data.
    assert split == "train"

    speech_dir = os.path.join(dataset_dir, "clean_{}set_wav".format(split))
    mixture_dir = os.path.join(dataset_dir, "noisy_{}set_wav".format(split))

    os.makedirs(hdf5s_dir, exist_ok=True)

    # Read names.
    audio_names = sorted(os.listdir(speech_dir))

    params = []

    for audio_index, audio_name in enumerate(audio_names):

        speech_path = os.path.join(speech_dir, audio_name)
        mixture_path = os.path.join(mixture_dir, audio_name)

        hdf5_path = os.path.join(
            hdf5s_dir, "{}.h5".format(pathlib.Path(audio_name).stem)
        )

        param = (
            audio_index,
            audio_name,
            speech_path,
            mixture_path,
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


def write_single_audio_to_hdf5(param: List) -> NoReturn:
    r"""Write single audio into hdf5 file."""

    (
        audio_index,
        audio_name,
        speech_path,
        mixture_path,
        mono,
        sample_rate,
        hdf5_path,
    ) = param

    with h5py.File(hdf5_path, "w") as hf:

        hf.attrs.create("audio_name", data=audio_name, dtype="S100")
        hf.attrs.create("sample_rate", data=sample_rate, dtype=np.int32)

        speech = load_audio(audio_path=speech_path, mono=mono, sample_rate=sample_rate)
        # speech: (channels_num, audio_samples)

        mixture = load_audio(
            audio_path=mixture_path, mono=mono, sample_rate=sample_rate
        )
        # mixture: (channels_num, audio_samples)

        noise = mixture - speech
        # noise: (channels_num, audio_samples)

        hf.create_dataset(name='speech', data=float32_to_int16(speech), dtype=np.int16)
        hf.create_dataset(name='noise', data=float32_to_int16(noise), dtype=np.int16)

    print('{} Write hdf5 to {}'.format(audio_index, hdf5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory of the Voicebank-Demand dataset.",
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

    # Pack audios into hdf5 files.
    pack_audios_to_hdf5s(args)
