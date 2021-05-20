import argparse
import os
import time
import pickle

import h5py
import librosa
import musdb
import numpy as np

from music_source_separation.utils import float32_to_int16, read_yaml


# Source types of the MUSDB18 dataset.
SOURCE_TYPES = ["vocals", "drums", "bass", "other", "accompaniment"]


def pack_audios_to_hdf5s(args):
    r"""Pack (resampled) audio files into hdf5 files to speed up loading."""

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

    # Load dataset according to subset and split.
    mus = musdb.DB(root=dataset_dir, subsets=[subset], split=split)

    print("Subset: {}, Split: {}, Total pieces: {}".format(subset, split, len(mus)))

    # Paths
    os.makedirs(hdf5s_dir, exist_ok=True)

    pack_hdf5s_time = time.time()

    # Traverse all tracks.
    for track_index, track in enumerate(mus.tracks):
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
            audio = preprocess_audio(
                audio, mono, track.rate, sample_rate, resample_type
            )
            # (channels_num, audio_samples) | (audio_samples,)

            hf.create_dataset(
                name="mixture", data=float32_to_int16(audio), dtype=np.int16
            )

        print("{} Write to {}, {}".format(track_index, hdf5_path, audio.shape))

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


def preprocess_audio(audio, mono, origin_sr, sr, resample_type):
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
    # (channels_num, audio_samples) | (audio_samples,)

    return output


def create_indexes(args):
    r"""Create and write out training indexes into disk. In training a source
    separation system, training indexes will be shuffled and iterated for
    selecting segments to be mixed. E.g., the training indexes_dict looks like:

        {'vocals': [
             [./piece1.h5, 0, 132300],
             [./piece1.h5, 4410, 136710],
             [./piece1.h5, 8820, 141120],
             ...
         ],
         'accompaniment': [
             [./piece1.h5, 0, 132300],
             [./piece1.h5, 4410, 136710],
             [./piece1.h5, 8820, 141120],
             ...
         ]
        }
    """

    # Arugments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml

    # Only create indexes for training, because evalution is on entire pieces.
    split = "train"

    # Read config file.
    configs = read_yaml(config_yaml)

    sample_rate = configs["sample_rate"]
    segment_samples = int(configs["segment_seconds"] * sample_rate)

    # Path to write out index.
    indexes_path = os.path.join(workspace, configs[split]["indexes"])
    os.makedirs(os.path.dirname(indexes_path), exist_ok=True)

    source_types = configs[split]["source_types"].keys()
    # E.g., ['vocals', 'accompaniment']

    indexes_dict = {source_type: [] for source_type in source_types}
    # E.g., indexes_dict will looks like: {
    #     'vocals': [
    #         [./piece1.h5, 0, 132300],
    #         [./piece1.h5, 4410, 136710],
    #         [./piece1.h5, 8820, 141120],
    #         ...
    #     ],
    #     'accompaniment': [
    #         [./piece1.h5, 0, 132300],
    #         [./piece1.h5, 4410, 136710],
    #         [./piece1.h5, 8820, 141120],
    #         ...
    #     ]
    # }

    # tmp_dict = {source_type: {} for source_type in source_types}

    # Get training indexes for each source type.
    for source_type in source_types:

        print("--- {} ---".format(source_type))

        dataset_types = configs[split]["source_types"][source_type]
        # E.g., ['musdb18', ...]

        # Each source can come from mulitple datasets.
        for dataset_type in dataset_types:

            hdf5s_dir = os.path.join(
                workspace, dataset_types[dataset_type]["directory"]
            )
            hop_samples = int(dataset_types[dataset_type]["hop_seconds"] * sample_rate)

            hdf5_names = os.listdir(hdf5s_dir)
            print("Hdf5 files num: {}".format(len(hdf5_names)))

            # Traverse all packed hdf5 files of a dataset.
            for n, hdf5_name in enumerate(hdf5_names):

                print(n, hdf5_name)
                hdf5_path = os.path.join(hdf5s_dir, hdf5_name)

                with h5py.File(hdf5_path, "r") as hf:

                    start_sample = 0
                    while start_sample + segment_samples < hf[source_type].shape[-1]:
                        indexes_dict[source_type].append(
                            [hdf5_path, start_sample, start_sample + segment_samples]
                        )

                        start_sample += hop_samples

                # tmp_dict[source_type][hdf5_path] = [0, start_sample - segment_samples]

        print(
            "Total indexes for {}: {}".format(
                source_type, len(indexes_dict[source_type])
            )
        )

    pickle.dump(indexes_dict, open(indexes_path, "wb"))
    print("Write index dict to {}".format(indexes_path))

    # pickle.dump(tmp_dict, open('tmp.pkl', "wb"))
    # print("Write to {}".format('tmp.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # Pack audios to hdf5 files.
    parser_pack_audios = subparsers.add_parser("pack_audios_to_hdf5s")
    parser_pack_audios.add_argument(
        "--dataset_dir", type=str, required=True, help="Directory of the MUSDB18 dataset."
    )
    parser_pack_audios.add_argument(
        "--subset",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Train subset: 100 pieces; test subset: 50 pieces.",
    )
    parser_pack_audios.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["", "train", "valid"],
        help="Use '' to use all 100 pieces to train. Use 'train' to use 86 \
            pieces for train, and use 'test' to use 14 pieces for valid.",
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

    # Create training indexes.
    parser_create_indexes = subparsers.add_parser("create_indexes")
    parser_create_indexes.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_create_indexes.add_argument(
        "--config_yaml", type=str, required=True, help="User defined config file."
    )

    # Parse arguments.
    args = parser.parse_args()

    if args.mode == "pack_audios_to_hdf5s":
        pack_audios_to_hdf5s(args)

    elif args.mode == "create_indexes":
        create_indexes(args)

    else:
        raise Exception("Incorrect arguments!")
